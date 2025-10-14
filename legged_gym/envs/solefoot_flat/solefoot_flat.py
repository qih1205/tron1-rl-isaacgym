# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float, CubicSpline
)
from .solefoot_flat_config import BipedCfgSF

import math
from time import time
from warnings import WarningMessage
import numpy as np
import os
from typing import Tuple, Dict
import random


class BipedSF(BaseTask):
    def __init__(
        self, cfg: BipedCfgSF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        
        # 初始化奖励函数用的高度观测值
        self.reward_points_x = self.cfg.terrain.reward_measure_point_x
        self.reward_points_y = self.cfg.terrain.reward_measure_point_y
        self.reward_heights = 0  # 初始化为0，后续会在_post_physics_step_callback中更新
        
        self.init_done = True

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)

        self.base_lin_acc = (self.base_lin_vel - self.last_base_lin_vel) / self.dt
        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, self.base_lin_acc)

        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.dof_pos_int += (self.dof_pos - self.raw_default_dof_pos) * self.dt
        self.power = torch.abs(self.torques * self.dof_vel)

        # self.dof_jerk = (self.last_dof_acc - self.dof_acc) / self.dt

        self.compute_foot_state()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        self._post_physics_step_callback()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # self.last_dof_acc[:] = self.dof_acc[:]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_positions[:] = self.foot_positions[:]

    def compute_foot_state(self):
        super().compute_foot_state()
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

    def compute_observations(self):
        """Computes observations"""
        proprioceptive_obs, critic_obs_buf_base = self.compute_self_observations()
        
        # Update observation history with proprioceptive observations only (no heights)
        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs :], proprioceptive_obs), dim=-1
        )
        
        # Build complete observation buffer
        self.obs_buf = proprioceptive_obs.clone()
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
            # Also add heights to critic observations
            self.critic_obs_buf = torch.cat((critic_obs_buf_base, heights), dim=-1)
        else:
            self.critic_obs_buf = critic_obs_buf_base

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        # Calculate the correct size for noise vector
        noise_vec_size = self.num_obs
        if self.cfg.terrain.measure_heights:
            noise_vec_size += self.cfg.env.num_height_samples
        
        noise_vec = torch.zeros(noise_vec_size, device=self.device, dtype=torch.float)
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:14] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[14:22] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[22:36] = 0.0  # previous actions, clock inputs, gaits
        # Add noise for height measurements if enabled
        if self.cfg.terrain.measure_heights:
            noise_vec[36:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements if hasattr(noise_scales, 'height_measurements') else 0.0
        return noise_vec

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR
        )
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        #gym.get_asset_dof_properties返回一个包含关节物理属性的命名数组
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)#从asset中获取机器人全部关节的名称，body_names是一个列表，列表中的每个元素是一个字符串，表示关节的名称
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)#从asset中获取自由度名称，前面加self说明是一个成员，会在类的其他方法中使用
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]#在body_names中找到所有包含foot的关节名称
        contact_names = []
        if hasattr(self.cfg.asset, "contact_name"):#如果body_names中包含contact_name，则将该关节名称添加到contact_names列表中
            contact_names = [s for s in body_names if self.cfg.asset.contact_name in s]
        penalized_contact_names = []#定义：这些关节如果接触，会受到惩罚
        for name in self.cfg.asset.penalize_contacts_on:
            #遍历penalize_contacts_on列表，将列表中的每个元素在body_names中搜索，如果找到，则将该关节名称添加到penalized_contact_names列表中
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []#定义：这些关节如果接触，会终止
        for name in self.cfg.asset.terminate_after_contacts_on:
            #遍历terminate_after_contacts_on列表，将列表中的每个元素在body_names中搜索，如果找到，则将该关节名称添加到termination_contact_names列表中
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (#将init_state的pos、rot、lin_vel、ang_vel拼接成一个列表
            self.cfg.init_state.pos#位置
            + self.cfg.init_state.rot#旋转
            + self.cfg.init_state.lin_vel#线速度
            + self.cfg.init_state.ang_vel#角速度
        )
        self.base_init_state = to_torch(#将base_init_state_list转换为torch张量，并存储到self.base_init_state中
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()#创建一个gymapi.Transform对象，并存储到start_pose中
        #gymapi.Transform 包含两个主要属性：
        #p：位置向量，类型为 gymapi.Vec3，表示在三维空间中的平移
        #r：旋转四元数，类型为 gymapi.Quat，表示在三维空间中的旋转
        #示例：
        #transform = gymapi.Transform()
        #transform.p = gymapi.Vec3(1.0, 0.0, 0.5)  # x=1.0, y=0.0, z=0.5
        #transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 单位四元数（无旋转）
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])#将base_init_state的前3个元素作为位置存储到start_pose中，设置机器人的初始位置

        self._get_env_origins()#设置环境（plane， heightfield， trimesh）的初始位置
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        self.friction_coef = torch.zeros(#创建全0的张量，存储摩擦系数
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.restitution_coef = torch.zeros(#创建全0的张量，存储恢复系数
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_mass = torch.zeros(#创建全0的张量，存储基座质量
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.whole_body_mass = torch.zeros(#创建全0的张量，存储整个身体质量
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.base_com = torch.zeros(#创建全0的张量，形状为 (num_envs, 3)，3维坐标 (x, y, z)，存储基座质心
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                #gym.create_env 是 Isaac Gym 中用于创建仿真环境的核心函数。它负责在
                #仿真器中创建一个独立的仿真环境实例。
                #env_ptr = gym.create_env(sim, lower, upper, num_per_row)
                #通过 gym.create_sim() 创建的仿真器对象，包含物理引擎和渲染上下文
                #lower：定义环境在三维空间中的最小范围坐标
                #upper：定义环境在三维空间中的最大范围坐标
                #num_per_row：当创建多个环境时，指定网格布局中每行的环境数量
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            #env_origins 是包含所有环境初始位置的列表。每个元素是一个 3 维向量，表示对应环境在三维空间中的初始位置。 
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            #pos[:2] 表示取 pos 的前两个元素
            #+= 表示将生成的随机浮点数添加到 pos 的前两个元素上
            #torch_rand_float 是用于生成随机浮点数的函数。
            #-1.0 和 1.0 是随机浮点数的范围
            #(2, 1) 是形状参数，表示生成一个 2x1 的张量
            #device=self.device 指定生成的张量存储在哪个设备上
            #squeeze(1) 用于从张量中移除单维度的维度
            start_pose.p = gymapi.Vec3(*pos)
            #gymapi.Vec3(*pos) 将 pos 转换为 gymapi.Vec3 类型，并存储到 start_pose 中
            #实现为每个环境的初始位置添加随机偏移

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            #为每个环境分配不同的摩擦系数和恢复系数。

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            #将上一行中处理后的属性更新到robot_asset上。

            actor_handle = self.gym.create_actor(
                #功能是在指定的环境中创建一个机器人
                #函数返回一个 actor_handle，这是角色的唯一标识符，后续所有对该角色的操作
                #如设置关节属性、获取状态等都需要使用这个句柄。
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            # cartpole_handle = self.gym.create_actor(
            #     env_ptr,                    # 环境指针
            #     cartpole_asset,             # 预加载的cartpole资源
            #     pose,                       # 初始位姿，
            #     "cartpole",                 # 角色名称，
            #     i,                          # 环境索引作为分组ID，
            #     1,                          # 碰撞过滤掩码，
            #     0                           # 分割ID，
            # )
            dof_props = self._process_dof_props(dof_props_asset, i)
            #为每个环境设置不同的关节参数
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            #将上一行中处理后的关节参数同步更新到环境和机器人上。
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )#读取刚体属性并存入body_props
            body_props = self._process_rigid_body_props(body_props, i)#修改刚体属性
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )#将修改后的刚体属性更新到环境和机器人上，并重新计算惯性张量
            self.envs.append(env_handle)#
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        self.contact_indices = torch.zeros(
            len(contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )
        for i in range(len(contact_names)):
            self.contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], contact_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._check_walk_stability(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_self_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        n = env_ids.size(0)
        indices = torch.randperm(n)

        half_size = n // 2
        half_indices = indices[:half_size]
        remaining_indices = indices[half_size:]

        half_list = env_ids[half_indices]
        remaining_list = env_ids[remaining_indices]
        self.dof_pos[half_list] = self.default_dof_pos[half_list, :] + torch_rand_float(
            -0.5, 0.5, (len(half_list), self.num_dof), device=self.device
        )
        self.dof_pos[remaining_list] = self.init_stand_dof_pos[remaining_list, :] + torch_rand_float(
            -0.5, 0.5, (len(remaining_list), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale, # 5 commands
            self.critic_obs_buf
        )

    def compute_self_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        # compute critic_obs_buf
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, obs_buf), dim=-1)
        return obs_buf, critic_obs_buf

    def get_observations(self):
        # Return full observations (with heights) for storage
        # Actor will extract proprioceptive part (first 36 dims)
        return (
            self.obs_buf,
            self.obs_history,
            self.commands[:, :5] * self.commands_scale, # 5 commands
            self.critic_obs_buf
        )

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                    self.episode_length_buf
                    % int(self.cfg.commands.resampling_time / self.dt)
                    == 0
            )
                .nonzero(as_tuple=False)
                .flatten()
        )
        self._resample_commands(env_ids, False)
        self._resample_gaits(env_ids)
        self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 1.0 * wrap_to_pi(self.commands[:, 5] - heading)

        self._resample_zero_commands(env_ids)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()
            self.reward_heights = self._get_heights(reward=True)
        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.reward_heights, dim=1
        )

    def _step_contact_targets(self):
        super()._step_contact_targets()
        self._generate_des_ee_ref()

    def _generate_des_ee_ref(self):
        frequencies = self.gaits[:, 0]
        mask_0 = (self.gait_indices < 0.25) & (self.gait_indices >= 0.0)  # lift up
        mask_1 = (self.gait_indices < 0.5) & (self.gait_indices >= 0.25)  # touch down
        mask_2 = (self.gait_indices < 0.75) & (self.gait_indices >= 0.5)  # lift up
        mask_3 = (self.gait_indices <= 1.0) & (self.gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.cfg.gait.touch_down_vel
        swing_end_vel[mask_3] = self.cfg.gait.touch_down_vel

        # generate desire foot z trajectory
        swing_height = self.gaits[:, 3]
        # self.des_foot_height = 0.5 * swing_height * (1 - torch.cos(4 * np.pi * self.gait_indices))
        # self.des_foot_velocity_z = 2 * np.pi * swing_height * frequencies * torch.sin(
        #     4 * np.pi * self.gait_indices)

        start = {'time': swing_start_time, 'position': swing_start_pos * swing_height,
                 'velocity': torch.zeros(self.num_envs, device=self.device)}
        end = {'time': swing_end_time, 'position': swing_end_pos * swing_height,
               'velocity': swing_end_vel}
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(self.gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(self.gait_indices / frequencies)

    def _resample_gaits(self, env_ids):
        super()._resample_gaits(env_ids)
        self._resample_stand_still_gait_commands(env_ids)

    def _check_walk_stability(self, env_ids):
        if len(env_ids) != 0:
            self.mean_episode_len = torch.mean(self.episode_length_buf[env_ids].float(), dim=0).cpu().item()
        if self.mean_episode_len > 950:
            self.stable_episode_length_count += 1
            # print("Stable Episode Length:{}, count:{}.".format(self.mean_episode_len, self.stable_episode_length_count))
        else:
            self.stable_episode_length_count = 0

    def _resample_commands(self, env_ids, is_start=True):
        """Randommly select commands of some environments

                Args:
                    env_ids (List[int]): Environments ids for which new commands are needed
                """
        self.commands[env_ids, 0] = (self.command_ranges["lin_vel_x"][env_ids, 1]
                                     - self.command_ranges["lin_vel_x"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["lin_vel_x"][env_ids, 0]
        self.commands[env_ids, 1] = (self.command_ranges["lin_vel_y"][env_ids, 1]
                                     - self.command_ranges["lin_vel_y"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["lin_vel_y"][env_ids, 0]
        self.commands[env_ids, 2] = (self.command_ranges["ang_vel_yaw"][env_ids, 1]
                                     - self.command_ranges["ang_vel_yaw"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["ang_vel_yaw"][env_ids, 0]
        self.commands[env_ids, 3] = (self.command_ranges["base_height"][env_ids, 1]
                                     - self.command_ranges["base_height"][env_ids, 0]) \
                                    * torch.rand(len(env_ids), device=self.device) \
                                    + self.command_ranges["base_height"][env_ids, 0]

        self._resample_stand_still_commands(env_ids, is_start)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 5] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

    def _resample_zero_commands(self, env_ids):
        thresh = 0.25
        indices_to_update = env_ids[(self.commands[env_ids, 0] < thresh) & (self.commands[env_ids, 0] > -thresh)]
        self.commands[indices_to_update, :3] = 0.0

    def _resample_stand_still_commands(self, env_ids, is_start=True):
        if (not self.walk_stability) and self.stable_episode_length_count >= 10:
            self.walk_stability = True
            self.stable_episode_length_count = 0
        if self.walk_stability and (not self.stand_still_stability) and self.stable_episode_length_count >= 10:
            self.stand_still_stability = True
        if self.walk_stability and (not self.stand_still_stability):
            if not is_start:
                indices_to_update = env_ids[self.commands[env_ids, 4] == 0]
                self.commands[indices_to_update, 4] = (self.command_ranges["stand_still"][indices_to_update, 1]
                                                       - self.command_ranges["stand_still"][indices_to_update, 0]) \
                                                      * torch.randint(0, 2, (len(indices_to_update),),
                                                                      device=self.device) \
                                                      + self.command_ranges["stand_still"][indices_to_update, 0]
                indices_to_update1 = indices_to_update[self.commands[indices_to_update, 4] == 1]
                self.commands[indices_to_update1, :3] = 0.0
            else:
                self.commands[env_ids, 4] = 0
        elif self.walk_stability and self.stand_still_stability:
            self.commands[env_ids, 4] = (self.command_ranges["stand_still"][env_ids, 1]
                                         - self.command_ranges["stand_still"][env_ids, 0]) \
                                        * torch.randint(0, 2, (len(env_ids),), device=self.device) \
                                        + self.command_ranges["stand_still"][env_ids, 0]
            indices_to_update = env_ids[self.commands[env_ids, 4] == 1]
            self.commands[indices_to_update, :3] = 0.0
        else:
            self.commands[env_ids, 4] = 0

    def _resample_stand_still_gait_commands(self, env_ids):
        # indices_to_update = env_ids[self.commands[env_ids, 4] == 1]
        # self.gaits[indices_to_update, :] = 0.0
        pass

    def _resample_stand_still_gait_clock(self):
        indices_to_update = torch.nonzero(self.commands[:, 4] == 1).squeeze()
        gait_indices = self.gait_indices[indices_to_update]

        mask_0_5_to_0_55 = (gait_indices >= 0.5) & (gait_indices < 0.55)
        mask_0_0_to_0_05 = (gait_indices >= 0.0) & (gait_indices < 0.05)
        mask_0_95_to_1_0 = (gait_indices >= 0.95) & (gait_indices < 1.0)

        self.gait_indices[indices_to_update[mask_0_5_to_0_55]] = 0.5
        self.gait_indices[indices_to_update[mask_0_0_to_0_05]] = 0.0
        self.gait_indices[indices_to_update[mask_0_95_to_1_0]] = 0.0

        mask_else = ~(mask_0_5_to_0_55 | mask_0_0_to_0_05 | mask_0_95_to_1_0)
        self.commands[indices_to_update[mask_else], 4] = 0

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_heights = torch.zeros_like(self.foot_positions[:, :, 2])
        self.last_base_lin_vel = self.base_lin_vel.clone()

        self.base_lin_acc = torch.zeros_like(self.base_lin_vel)
        self.variances_per_env = 0
        self.init_stand_dof_pos = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            if hasattr(self.cfg.init_state, "init_stand_joint_angles"):
                stand_angle = self.cfg.init_state.init_stand_joint_angles[name]
                self.init_stand_dof_pos[:, i] = stand_angle

        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, 1, 1],
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["base_height"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["base_height"][:] = torch.tensor(
            self.cfg.commands.ranges.base_height
        )
        self.command_ranges["stand_still"] = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.command_ranges["stand_still"][:] = torch.tensor(
            self.cfg.commands.ranges.stand_still
        )

        self.des_foot_height = torch.zeros(self.num_envs,
                                           dtype=torch.float,
                                           device=self.device, requires_grad=False, ) # TODO
        self.des_foot_velocity_z = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                               requires_grad=False, ) # TODO

    def pre_physics_step(self):
        self.rwd_linVelTrackPrev = self._reward_tracking_lin_vel()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        self.rwd_orientationPrev = self._reward_orientation()
        # self.rwd_jointRegPrev = self._reward_joint_regularization()
        self.rwd_baseHeightPrev = self._reward_base_height()
        if "tracking_contacts_shaped_height" in self.reward_scales.keys():
            self.rwd_swingHeightPrev = self._reward_tracking_contacts_shaped_height()

    def sqrdexp(self, x):
        """ shorthand helper for squared exponential
        """
        return torch.exp(-torch.square(x) / self.cfg.rewards.tracking_sigma)
    
    # ----------------------rewards----------------------
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (
                    1
                    - torch.exp(
                        -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                    )
                )

        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (
                    1
                    - torch.exp(
                        -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                    )
                )
                # if self.cfg.terrain.mesh_type == "plane":
                swing_phase = 1 - desired_contact[:, i]
                reward += swing_phase * (1 - torch.exp(
                    -((self.foot_velocities[:, i, 2] - self.des_foot_velocity_z) ** 2)
                    / self.cfg.rewards.gait_vel_sigma)
                )
        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_tracking_contacts_shaped_height(self):
        foot_heights = self.foot_heights
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_height"] > 0:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * torch.exp(
                    -(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma)
        else:
            for i in range(len(self.feet_indices)):
                swing_phase = 1 - desired_contact[:, i]
                # if self.cfg.terrain.mesh_type == "plane":
                reward += swing_phase * (
                        1 - torch.exp(-(foot_heights[:, i] - self.des_foot_height) ** 2 / self.cfg.rewards.gait_height_sigma)
                )
                stand_phase = desired_contact[:, i]
                reward += stand_phase * (1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma))
        return torch.where(self.commands[:, 4] == 0, reward / len(self.feet_indices), 0)

    def _reward_feet_distance(self):
        """惩罚双脚之间的水平距离过近
        
        这个奖励函数用于防止机器人双脚靠得太近而导致不稳定,
        确保机器人保持足够的支撑底面积。不同模式下使用不同的惩罚策略。
        
        Returns:
            torch.Tensor: 每个环境的惩罚值,形状为 (num_envs,)
                         双脚距离越小于最小距离,惩罚越大
        """
        # 1. 计算双脚在水平面(xy平面)上的距离
        # foot_positions[:, 0, :2] 是第一只脚的xy位置
        # foot_positions[:, 1, :2] 是第二只脚的xy位置
        # torch.norm 计算两点之间的欧几里得距离
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        
        # 2. 根据不同的模式返回不同的惩罚
        # 条件判断: 站立模式(commands[:, 4] == 1) 且 目标高度较低(commands[:, 3] <= 0.3)
        return torch.where(
            # 当机器人处于站立模式且蹲得较低时(高度≤0.3):
            # 使用双向惩罚: |实际距离 - 最小距离|
            # 这意味着既惩罚双脚太近,也惩罚双脚太远
            # 裁剪到[0, 1]范围内,避免过大的惩罚值
            torch.logical_and(self.commands[:, 4] == 1, self.commands[:, 3] <= 0.3),
            torch.clip(torch.abs(self.cfg.rewards.min_feet_distance - feet_distance), 0, 1),
            
            # 其他情况(行走模式或站立时高度>0.3):
            # 使用单向惩罚: max(0, 最小距离 - 实际距离)
            # 只惩罚双脚太近的情况,不惩罚双脚太远
            # 裁剪到[0, 1]范围内
            torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1),
        )

    def _reward_feet_regulation(self):
        """惩罚脚部在接近地面时的水平滑动
        
        这个奖励函数用于防止机器人的脚在接近或接触地面时出现水平方向的滑动,
        鼓励脚部在着地前减速并稳定接触,提高步态的稳定性和能量效率。
        当脚越接近地面,惩罚越大。
        
        Returns:
            torch.Tensor: 每个环境的惩罚值,形状为 (num_envs,)
                         脚部越低且水平速度越大,惩罚越大
        """
        # 1. 定义"接近地面"的高度阈值
        # 使用目标基座高度的2.5%作为参考高度
        # 例如: 如果base_height_target=0.4m, 则feet_height=0.01m
        feet_height = self.cfg.rewards.base_height_target * 0.025
        
        # 2. 计算惩罚值
        # 这个惩罚由两个因素的乘积组成:
        reward = torch.sum(
            # 因子1: exp(-foot_heights / feet_height)
            # 这是一个高度权重函数,当脚部高度接近0时,权重接近1
            # 当脚部离地面越远,权重呈指数衰减,接近0
            # 意味着只在脚部接近地面时进行惩罚
            torch.exp(-self.foot_heights / feet_height)
            
            # 因子2: ||foot_velocities_xy||^2
            # foot_velocities[:, :, :2] 是脚部在xy平面的速度(水平速度)
            # torch.norm 计算水平速度的大小
            # torch.square 计算速度的平方,使惩罚对大速度更敏感
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            
            # 对所有脚部求和(双足机器人有2只脚)
            dim=1,
        )
        
        # 3. 返回惩罚值
        # 当脚部接近地面且仍有较大的水平速度时,惩罚最大
        # 鼓励机器人在脚着地前减小水平速度,实现平稳着地
        return reward

    def _reward_power(self):
        # Penalize torques
        joint_array = [i for i in range(self.num_dof)]
        joint_array.remove(3)
        joint_array.remove(7)
        return torch.sum(torch.abs(self.torques[:, joint_array] * self.dof_vel[:, joint_array]), dim=1)

    def _reward_collision(self):
        reward = torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
            )            > 1.0,
            dim=1,
        )
        return reward

    def _reward_base_height(self):
        """惩罚机器人基座高度偏离目标值
        
        这个奖励函数用于控制机器人保持特定的基座高度,确保机器人在
        行走或站立时不会蹲得太低或站得太高。该函数在所有模式下都激活。
        
        Returns:
            torch.Tensor: 每个环境的惩罚值,形状为 (num_envs,)
                         偏离目标高度越大,惩罚越大
        """
        # 1. 计算基座相对于地形的实际高度
        # root_states[:, 2] 是基座在世界坐标系下的z坐标
        # measured_heights 是脚下地形的高度采样点(多个采样点的平均值)
        # 两者相减并取平均值得到基座离地的实际高度
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.reward_heights, dim=1
        )
        
        # 2. 计算高度误差作为惩罚
        # cfg.rewards.base_height_target 是配置文件中设定的目标基座高度
        # 使用绝对值误差: |实际高度 - 目标高度|
        # 注释掉的平方误差: (实际高度 - 目标高度)^2 会对大偏差给予更严厉的惩罚
        # reward = torch.square(base_height - self.commands[:, 3])  # 使用命令高度(站立模式的动态目标)
        reward = torch.abs(base_height - self.cfg.rewards.base_height_target)  # 使用固定目标高度
        
        # 3. 返回惩罚值
        # 注释掉的代码: 在站立模式(commands[:, 4] == 1)时会给予1.5倍的惩罚
        # 当前实现: 对所有模式(行走和站立)统一使用相同的惩罚
        # return torch.where(self.commands[:, 4] == 0, reward, reward * 1.5)
        return reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_ankle_torque_limits(self):
        torque_limit = torch.cat((self.torque_limits[3].view(1) * self.cfg.rewards.soft_torque_limit,
                                  self.torque_limits[7].view(1) * self.cfg.rewards.soft_torque_limit),
                                 dim=-1, )
        torque = torch.cat((self.torques[:, 3].view(self.num_envs, 1),
                            self.torques[:, 7].view(self.num_envs, 1)), dim=-1)
        return torch.sum(
            torch.pow(torque / torque_limit, 8),
            dim=1,
        )

    def _reward_relative_feet_height_tracking(self):
        """奖励机器人在站立模式下跟踪双脚相对于身体的目标高度
        
        这个奖励函数用于控制机器人在站立状态时保持特定的脚部高度,
        可以实现蹲起等动作。只在站立模式(commands[:, 4] == 1)时激活。
        
        Returns:
            torch.Tensor: 每个环境的奖励值,形状为 (num_envs,)
                         误差越小奖励越高,最大值为1(完全匹配)
        """
        # 1. 计算基座相对于地形的高度
        # root_states[:, 2] 是基座在世界坐标系下的z坐标
        # measured_heights 是脚下地形的高度采样点
        # 两者相减并取平均值得到基座离地高度
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.reward_heights, dim=1
        )
        
        # 2. 计算双脚在身体坐标系下的高度
        # base_height - foot_heights = 脚相对于基座的高度(向上为正)
        # 形状: (num_envs, num_feet)
        feet_height_in_body_frame = base_height.view(self.num_envs, 1) - self.foot_heights
        
        # 3. 使用高斯函数将跟踪误差转换为奖励
        # commands[:, 3] 是目标的双脚相对高度命令
        # 计算双脚高度与目标高度的平方误差和
        # 通过 exp(-error/sigma) 将误差映射到 [0, 1] 范围
        # 误差为0时奖励为1,误差越大奖励越接近0
        reward = torch.exp(
            -torch.sum(
                torch.square(
                    feet_height_in_body_frame - self.commands[:, 3].view(self.num_envs, 1)
                ),
                dim=-1) / self.cfg.rewards.height_tracking_sigma
        )
        
        # 4. 只在站立静止模式时给予奖励
        # commands[:, 4] == 1 表示机器人处于站立模式
        # 其他模式(如行走)时奖励为0
        return torch.where(self.commands[:, 4] == 1, reward, 0)

    def _reward_zero_command_nominal_state(self):
        # Penalize the hip joint pos in zero command
        dof_pos = self.dof_pos - self.raw_default_dof_pos
        reward = torch.sum(
            torch.square(dof_pos[:, [1, 5]]), dim=1
        )
        return reward * torch.logical_and(torch.norm(self.commands[:, :3], dim=1) < 0.05, self.commands[:, 4] == 0)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_keep_ankle_pitch_zero_in_air(self):
        ankle_pitch = torch.abs(self.dof_pos[:, 3]) * ~self.contact_filt[:, 0] + torch.abs(
            self.dof_pos[:, 7]) * ~self.contact_filt[:, 1]
        return torch.exp(-torch.abs(ankle_pitch) / 0.2)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions[:, :, 0]
                + self.last_actions[:, :, 1]
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~(self.time_out_buf | self.edge_reset_buf)

    def _reward_fail(self):
        return self.fail_buf > 0

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        torque_limit = self.torque_limits * self.cfg.rewards.soft_torque_limit
        return torch.sum(
            torch.pow(self.torques / torque_limit, 8),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_stand_still(self):

        return torch.sum(self.foot_heights, dim=1) * (
            torch.norm(self.commands[:, :3], dim=1) < self.cfg.commands.min_norm
        )

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                self.contact_forces[:, self.feet_indices, 2]
                - self.base_mass.mean() * 9.8 / 2
            ).clip(min=0.0),
            dim=1,
        )