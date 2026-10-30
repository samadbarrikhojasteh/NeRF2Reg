# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations
import os
from math import isnan

from nerfstudio.soft_morph import SoftDilation, SoftErode
import math
import cv2
import pandas as pd
from numpy.array_api import float32
from torch import nn, enable_grad
import cv2 as cv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
import torchvision
import numpy as np
import torch
import trimesh
from natsort.utils import final_data_transform_factory
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from trimesh.path.packing import visualize
from triton.language import dtype

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.optimizer2 import MeshAligner, device, Deformation, Spline, nerfacto2opencv, readjson, Basis, Affine
from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings,MeshRenderer,
    MeshRasterizer, SoftPhongShader,SoftSilhouetteShader,SoftPhongShader,TexturesVertex,PointLights,BlendParams, HardFlatShader, HardGouraudShader, HardPhongShader,)
from pytorch3d.renderer.mesh.shading import interpolate_face_attributes

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj , save_obj
from pytorch3d.loss import point_mesh_distance
from pytorch3d import _C
from pytorch3d import transforms
from pytorch3d.transforms import so3_exp_map, euler_angles_to_matrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use('Agg')
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.utils.math import safe_normalize
import torch.nn.functional as F


@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""


class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.soft_erode = SoftErode()
        self.soft_dilation = SoftDilation()
        # self.mlp_deformation = MLP(
        #     in_dim=3,
        #     num_layers=6, #3
        #     layer_width=64, #64
        #     out_dim=3,
        #     activation=nn.ReLU(),
        #     # out_activation=nn.Sigmoid(),
        #     out_activation=nn.ReLU(), #None
        #     implementation="tcnn",
        # )
        # self.mlp_deformation_hash = MLPWithHashEncoding(
        #     num_levels=16, #16
        #     min_res=16, #16
        #     max_res=2048, #2048
        #     log2_hashmap_size=19, #19
        #     features_per_level=2, #2
        #     num_layers=2,  # 2
        #     layer_width=64, #64
        #     out_dim=3,
        #     activation=nn.ReLU(),  # nn.ReLU()
        #     out_activation=None,
        #     implementation="tcnn",
        # )

        # Differentiable transformation matrix
        # self.optim = Affine()        # for Affine transformation

        # self.optim = MeshAligner()   # for rigid transformation
        self.optim = Spline()        # for Basis deformation



        # self.def_matrix = Deformation()
        # self.basis = Basis()

        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/liver_real/p1/preop_to_NeRF_new.obj")                # P1 transformed by PCA
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/liver_real/p1/transformed_preop_ICP.obj")            # P1 transformed by ICP

        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/liver_real/20180905/preop_to_NeRF_new.obj")          # P2 transformed by PCA
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/liver_real/20180905/transformed_preop_ICP.obj")      # P2 transformed by ICP

        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/liver_real/P5/preop_to_NeRF.obj")                    # P5 transformed by PCA
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/liver_real/P5/transformed_preop_ICP.obj")            # P5 transformed by ICP


        # def repair_obj_face_indices(input_path, output_path):
        #     with open(input_path, "r", encoding="utf-8") as src, \
        #             open(output_path, "w", encoding="utf-8") as dst:
        #
        #         for line in src:
        #             if line.lstrip().startswith("f "):
        #                 parts = line.strip().split()
        #                 repaired_tokens = []
        #
        #                 for token in parts[1:]:
        #                     indices = token.split("/")
        #
        #                     if len(indices) == 3 and indices[2] == "":
        #                         if indices[1] == "":
        #                             token = indices[0]  # v// -> v
        #                         else:
        #                             token = f"{indices[0]}/{indices[1]}"  # v/vt/ -> v/vt
        #
        #                     repaired_tokens.append(token)
        #
        #                 indentation = line[:len(line) - len(line.lstrip())]
        #                 line = indentation + " ".join([parts[0]] + repaired_tokens) + "\n"
        #
        #             dst.write(line)
        #
        # repaired_path = "/home/geintra/MIS/data/liver/liver_real/20180905/repaired.obj"
        # repair_obj_face_indices(dir, repaired_path)


        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/nasel/P42/transformed_nose_3_closed_4.obj")                # nose transformed by procrustes
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/nasel/P42/Maxillary.obj")

        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/uterus/preop_to_NeRF_new.obj")                             # Uterus transformed by PCA
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/uterus/transformed_preop_ICP.obj")                         # Uterus transformed by ICP

        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/new_rigid/preop_to_NeRF.obj")                        # synth_static transformed by PCA
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/new_rigid/transformed_preop_ICP.obj")                # synth_static transformed by ICP

        self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/deformation_NR1/preop_to_NeRF.obj")                    # synth_NR1 transformed by PCA
        # self.verts, self.faces = self.get_preop("/home/geintra/MIS/data/liver/deformation_NR1/transformed_preop_ICP.obj")              # synth_NR1 transformed by ICP

        # Compute center of the mass
        # center = torch.mean(verts, dim=0, keepdim=True)
        # mesh_center = (verts - center) * 0.03

        # self.lengths_org = self.compute_unique_edge_lengths(self.verts, self.faces.verts_idx)
        self.cosines_org = self.compute_face_angle_cosines(self.verts, self.faces.verts_idx, eps=1e-8)

        # scale = torch.linalg.norm(R[0,:,:], axis=0)
        # R_eul = transforms.matrix_to_euler_angles(R, convention="ZYX")  # [ 1.9140,  0.2681, -0.1895]
        # R_eul[0,0] = R_eul[0,0] * -1 #openCV to OpenGL ***This is our Euler rotation***  # [ -1.9140,  0.2681, -0.1895]
        # #
        # RR = transforms.euler_angles_to_matrix(R_eul, 'ZYX').to(device)
        # # Apply the scale to the rotation component only
        # RR_scale = RR * 1/0.03
        #
        # # Create a 4x4 transformation matrix
        # T = torch.eye(4, device=device)
        # T[:3, :3] = RR_scale
        # T[:3, 3] = torch.tensor([0.0,-150.0, 0.0]).to(device)
        # T_inv = torch.linalg.inv(T)
        # ones = torch.ones((verts.shape[0], 1), device=device)
        # homogeneous_preop = torch.cat([verts, ones], dim=1)
        # # Apply the transformation
        # preop_mesh_hemo = homogeneous_preop @ torch.transpose(T_inv, 0, 1)
        #
        # save_obj("transformed_preop.obj",preop_mesh_hemo[:,0:3],self.faces.verts_idx)

        # self.loss_log_registration = []
        self.loss_log_nerf = []     # list of dicts


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        param_groups["spline_matrix"] = list(self.optim.parameters())  # Add spline initial
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_only_depth(self, ray_bundle: RayBundle, only_depth= True):

        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        outputs = {}
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)

        outputs["pred_normals"] = pred_normals
        # outputs["pred_normals"] = self.normals_shader(pred_normals)
        outputs["expected_depth"] = expected_depth  # depth

        return outputs

    def get_outputs(self, ray_bundle: RayBundle, only_depth= False):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        # Smoothing on the depth vectors 1e-6 (the best performance)
        self.loss_depth = self.split(expected_depth) * 1e-6
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def smoo_normal(self, normals):
        dx = normals[:, :-1, :] - normals[:, 1:, :]
        dy = normals[:-1, :, :] - normals[1:, :, :]
        smoo_loss = (dx ** 2).sum(-1).mean() + (dy ** 2).sum(-1).mean()
        return smoo_loss

    # Splitting two by two (central selected ray and one of the up, down, left , right randomly and subtracting each of the splitted couples
    def split(self, tensor):
        if len(tensor) > 1:
            # Reshape the tensor to (2048, 2) by squeezing the singleton dimension
            reshaped_tensor = tensor.squeeze().reshape(-1, 2)
            # Calculate the differences between every two consecutive elements
            differences = torch.abs(reshaped_tensor[:, 1] - reshaped_tensor[:, 0])
            loss_depth = torch.sum(differences)
        else:
            loss_depth = torch.sum(torch.tensor(tensor))
        return loss_depth

    def get_preop(self, dir):
        verts_preop, faces_preop, properties_preop = load_obj(dir,load_textures=False, create_texture_atlas=False, device=device)
        return verts_preop, faces_preop

    def dis_triangle(self, mesh, triangle_index, origin):
        '''
        This function computes the barycentric distance
        :param mesh: 3D mesh(preoperative)
        :param triangle_index: Selected (Intersection rays) triangle indexes
        :param origin: Origin of rays
        :return: distance: Distance between origin rays and central of the triangles
        '''
        triangle_vertices = mesh.faces_packed()[triangle_index]
        # Compute the center of the triangle
        centroid_triangle = mesh.verts_padded()[0,:,:][triangle_vertices]
        center_x = (centroid_triangle[:, 0, 0] + centroid_triangle[:, 1, 0] + centroid_triangle[:, 2, 0]) / 3
        center_y = (centroid_triangle[:, 0, 1] + centroid_triangle[:, 1, 1] + centroid_triangle[:, 2, 1]) / 3
        center_z = (centroid_triangle[:, 0, 2] + centroid_triangle[:, 1, 2] + centroid_triangle[:, 2, 2]) / 3
        all_center = torch.stack([center_x, center_y, center_z], axis=1)
        # Compute Euclidean distance
        distance = torch.sqrt(torch.sum((all_center - origin) ** 2, axis=1))
        return distance

    def deformation_mlp(self, verts_preop):
        # Deformation MLP
        # preop_mlp= self.mlp_deformation(verts_preop)
        preop_mlp= self.mlp_deformation(verts_preop)
        # Added to the preop vert
        preop_deformed = (preop_mlp + verts_preop).to(torch.float32)
        return preop_deformed

    def get_surface_normal_by_depth(self, depth, K=None):
        """
        depth: (h, w) of float, the unit of depth is meter
        K: (3, 3) of float, the depth camere's intrinsic
        """
        fx, fy = K[0,0], K[1,1]
        dz_dv, dz_du = torch.gradient(depth)  # u, v mean the pixel coordinate in the image
        # u*depth = fx*x + cx --> du/dx = fx / depth
        du_dx = fx / depth  # x is xyz of camera coordinate
        dv_dy = fy / depth

        dz_dx = dz_du * du_dx
        dz_dy = dz_dv * dv_dy
        # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
        normal_cross = torch.dstack((-dz_dx, -dz_dy, torch.ones_like(depth)))
        # normalize to unit vector
        normal_unit = normal_cross / torch.linalg.norm(normal_cross, axis=2, keepdims=True)
        # set default normal to [0, 0, 0]
        normal_unit[~torch.isfinite(normal_unit).all(2)] = torch.tensor([0.0, 0.0, 0.0], device=depth.device)
        return normal_unit

    # def soft_erode(self,x, k:int):
    #     return -F.max_pool2d(-x, k, stride=1, padding=k // 2)
    #
    # def soft_dilation(self, mask, core, iters:int):
    #     g = core
    #     for _ in range(iters):
    #         g = torch.min(
    #             mask,
    #             F.max_pool2d(g, 3, stride=1, padding=1)
    #         )
    #     return g

    import numpy as np
    import trimesh

    def apply_smooth_nonrigid_deformation(
            mesh,
            bend_x_deg=5.0,
            bend_y_deg=4.0,
            twist_deg=5.0,
            stretch_x=0.01,
            stretch_y=0.01,
    ):
        """
        Apply a smooth synthetic non-rigid deformation.

        Parameters
        ----------
        bend_x_deg : float
            Maximum bending angle around the Y axis.

        bend_y_deg : float
            Maximum bending angle around the X axis.

        twist_deg : float
            Maximum twisting angle around the Z axis.

        stretch_x : float
            Spatially varying stretching along X.
            0.01 = approximately 1%.

        stretch_y : float
            Spatially varying stretching along Y.
            0.01 = approximately 1%.

        Returns
        -------
        deformed_mesh : trimesh.Trimesh
            Deformed mesh.

        displacement : (N, 3) ndarray
            Vertex displacement field.
        """

        X = np.asarray(mesh.vertices).copy()

        # Center mesh
        center = X.mean(axis=0)
        Xc = X - center

        x = Xc[:, 0]
        y = Xc[:, 1]
        z = Xc[:, 2]

        # Normalize z to [-1, 1]
        z_n = 2.0 * (z - z.min()) / (z.max() - z.min()) - 1.0

        # --------------------------------------------------
        # 1. Spatially varying stretching
        # --------------------------------------------------

        sx = 1.0 + stretch_x * z_n
        sy = 1.0 + stretch_y * z_n

        x1 = sx * x
        y1 = sy * y
        z1 = z.copy()

        # --------------------------------------------------
        # 2. Bending around Y
        # --------------------------------------------------

        bend_x = np.deg2rad(bend_x_deg) * z_n

        cos_x = np.cos(bend_x)
        sin_x = np.sin(bend_x)

        x2 = cos_x * x1 + sin_x * z1
        z2 = -sin_x * x1 + cos_x * z1

        # --------------------------------------------------
        # 3. Bending around X
        # --------------------------------------------------

        bend_y = np.deg2rad(bend_y_deg) * z_n

        cos_y = np.cos(bend_y)
        sin_y = np.sin(bend_y)

        y2 = cos_y * y1 - sin_y * z2
        z3 = sin_y * y1 + cos_y * z2

        # --------------------------------------------------
        # 4. Spatially varying twist
        # --------------------------------------------------

        twist = np.deg2rad(twist_deg) * z_n

        cos_t = np.cos(twist)
        sin_t = np.sin(twist)

        x3 = cos_t * x2 - sin_t * y2
        y3 = sin_t * x2 + cos_t * y2

        # --------------------------------------------------
        # Final coordinates
        # --------------------------------------------------

        X_deformed = np.column_stack([
            x3,
            y3,
            z3
        ]) + center

        # Ground-truth displacement
        displacement = X_deformed - X

        # Create deformed mesh
        deformed_mesh = mesh.copy()
        deformed_mesh.vertices = X_deformed

        return deformed_mesh, displacement

    def raster(self, step, T, verts_preop, faces_preop, dir, nerfacto_depth, cam_idx, mask_img, max_iterations, nerfacto_normal, do_render:bool = False):

        # verts_preop, faces_preop, _ = load_obj("/home/geintra/Downloads/template_samad.obj")
        # center = verts_preop.mean(dim=0)
        # modified_verts = (verts_preop - center) @ torch.tensor(([1.0, 0.0, 0.2],
        #                                                         [0.0, 0.8, 0.0],
        #                                                         [0.0, 0.0, 1.0])).T + center
        # transformed_mesh_path = "/home/geintra/Downloads"
        # save_obj(f"{transformed_mesh_path}/preop_non_rigid_transformed.obj", modified_verts, faces_preop.verts_idx)
        #
        # original_mesh = trimesh.load("/home/geintra/Downloads/template_samad.obj")
        # deformed_mesh = trimesh.load("/home/geintra/Downloads/template_samad_deformed.obj")
        #
        # deformed_mesh, displacement = apply_smooth_nonrigid_deformation(
        #     original_mesh,
        #     bend_x_deg=3,
        #     bend_y_deg=3,
        #     twist_deg=5,
        #     stretch_x=0.01,
        #     stretch_y=0.01
        # )
        #
        # transformed_mesh_path = "/home/geintra/Downloads"
        # deformed_mesh.export(f"{transformed_mesh_path}/synthetic_nonrigid_preop.obj")

######################### Apply the Affine transformation (12 parameters) ############################################
        # Convert points to homogeneous coordinates
        if T.shape[0] == 3: #[3,4]
            ones = torch.ones((verts_preop.shape[0], 1), device=verts_preop.device)
            homogeneous_preop = torch.cat([verts_preop, ones], dim=1)
            T_eye = torch.eye(4, device=verts_preop.device)
            T_eye[:3, :3] = T[:3, :3]
            T_eye[:3, 3] = T[:3, 3]
            T_inv = torch.linalg.inv(T_eye)
            preop_mesh_hemo = homogeneous_preop @ torch.transpose(T_inv, 0, 1)

        ## save_obj("preop_mesh_transformed_test.obj", preop_mesh_hemo[:, 0:3], self.faces.verts_idx)

############################# BASIS DEFORMATION ###########################################
        if T.shape[1] == 1: # [N,1]

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P1_4_PCA.npy").astype(np.float32) # P1 basis in Nerf coordinates (4)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P1_8_PCA.npy").astype(np.float32) # P1 basis in Nerf coordinates (8)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P1_16_PCA.npy").astype(np.float32)  # P1 basis in Nerf coordinates (16)_PCA

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P2_4_PCA.npy").astype(np.float32)  # P2 basis in Nerf coordinates (4)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P2_8_PCA.npy").astype(np.float32)  # P2 basis in Nerf coordinates (8)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P2_16_PCA.npy").astype(np.float32)  # P2 basis in Nerf coordinates (16)_PCA

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P5_4_PCA.npy").astype(np.float32)  # P5 basis in Nerf coordinates (4)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P5_8_PCA.npy").astype(np.float32)  # P5 basis in Nerf coordinates (8)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_P5_16_PCA.npy").astype(np.float32)  # P5 basis in Nerf coordinates (16)_PCA

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_nose_4.npy").astype(np.float32)    # nose basis in Nerf coordinates (4)
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_nose_8.npy").astype(np.float32)    # nose basis in Nerf coordinates (8)
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_nose_16.npy").astype(np.float32)   # nose basis in Nerf coordinates (16)

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_uterus_4_PCA.npy").astype(np.float32)   # Uterus basis in Nerf coordinates (4)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_uterus_8_PCA.npy").astype(np.float32)   # Uterus basis in Nerf coordinates (8)_PCA
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_uterus_16_PCA.npy").astype(np.float32)  # Uterus basis in Nerf coordinates (16)_PCA

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_synth_4_non_rigid.npy").astype(np.float32)              # synth_static basis in Nerf coordinates (4)
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_synth_8_non_rigid.npy").astype(np.float32)              # synth_static basis in Nerf coordinates (8)
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_synth_16_non_rigid.npy").astype(np.float32)             # synth_static basis in Nerf coordinates (16)

            # basis_load = np.load("/home/geintra/MIS/basis_nerf_synth_4_NR1.npy").astype(np.float32)              # synth_NR1 basis in Nerf coordinates (4)
            # basis_load = np.load("/home/geintra/MIS/basis_nerf_synth_8_NR1.npy").astype(np.float32)              # synth_NR1 basis in Nerf coordinates (8)
            basis_load = np.load("/home/geintra/MIS/basis_nerf_synth_16_NR1.npy").astype(np.float32)             # synth_NR1 basis in Nerf coordinates (16)

            basis = torch.from_numpy(basis_load).to(device=verts_preop.device)
            vec = torch.reshape(verts_preop.T, (-1, 1))
            X1 = (basis.T @ vec).to(torch.float32)
            X = X1 + T  # The T is basis vector [N,1]
            verts_preop_deformed = (basis @ X).to(torch.float32)
            preop_mesh_hemo = torch.reshape(verts_preop_deformed, (3, -1)).T

############################# RIGID TRANSFORMATION (7 parameters) ###########################################
        # Convert points to homogeneous coordinates
        if T.shape[0] == 4: # [4,4]
            ones = torch.ones((verts_preop.shape[0], 1), device=verts_preop.device)
            homogeneous_preop = torch.cat([verts_preop, ones], dim=1)
            T_inv = torch.linalg.inv(T)
            preop_mesh_hemo = (homogeneous_preop @ torch.transpose(T_inv, 0, 1)).to(torch.float32)
        ## save_obj("preop_before_basis.obj", preop_mesh_hemo[:, 0:3], self.faces.verts_idx) # Save the transferred preop model (using procrustes initialization)
##################################################################################################

        if do_render:
            # source_path = str(intraop_path)
            # intra_verts, intra_faces, _ = load_obj(source_path, load_textures=True, create_texture_atlas=True,device=device)
            # intra_mesh = Meshes(verts=[intra_verts], faces=[intra_faces.verts_idx])

            if os.path.exists(str(dir / "mask_val_ERODE")):
                test_mask_name = np.stack(sorted(os.listdir(dir / "mask_val_ERODE")))[cam_idx]
            else:
                raise Exception("No test masks found in the directory")
        else:
            # compute face angle every iteration over training
            # lengths_d = self.compute_unique_edge_lengths(preop_mesh_hemo, faces_preop)
            # self.loss_lengths = torch.norm(self.lengths_org - lengths_d)
            cosines = self.compute_face_angle_cosines(preop_mesh_hemo, faces_preop, eps=1e-8)
            self.loss_angle = ((self.cosines_org - cosines) ** 2).mean()
            # save_obj("M.obj", M[:,:3], faces_preop)
            # Get only training masks then extract the image's name
            if os.path.exists(str(dir / "mask_train_ERODE")):
                train_mask_name = np.stack(sorted(os.listdir(dir / "mask_train_ERODE")))[cam_idx]
            else:
                raise Exception("No masks found in the directory")

        # Get K and c2w from Json file
        frame_name_total, K, distortion, c2w, wh = readjson(dir)

        # sample_train_image = cv2.imread(str(dir/"mask_train_ERODE"/train_mask_name))
        img_height, img_width, _ = mask_img.shape # get rendered depth dimension
        # img_height, img_width, _ = depth_img.shape # get rendered depth dimension
        # compute scale factor between rendered depth image dimension and camera motion info(transformation JSON file)
        scale_h = wh["h"] / img_height
        scale_w = wh["w"] / img_width
        assert scale_h == scale_w
        scale = scale_h
        scale_depth = nerfacto_depth.shape[0] / img_height
        scale_normal = nerfacto_normal.shape[0] / img_height
        # print(scale)
        # Apply NeRFacto modifications to the c2w then select the corresponding w2c
        w2c = nerfacto2opencv(c2w)
        for d in w2c:
            if do_render:
                if d["name"] == os.path.join("images", test_mask_name):
                    w2c_selected = d["frame"]
            else:
                if d["name"] == os.path.join("images", train_mask_name):
                    w2c_selected = d["frame"]

        # modified K wrt image's downscale
        K[0:2,:] = K[0:2,:] / scale
        K_base = torch.tensor(np.append(K, [np.array([0.0, 0.0, 0.0, 1.0])], axis=0), dtype=torch.float32,
                              device=mask_img.device)
        K_base = torch.unsqueeze(K_base, dim=0)

        # Get normal surface from expected_depth
        # normal_by_depth = self.get_surface_normal_by_depth(depth = nerfacto_depth[:,:,0], K = K)

        R = torch.from_numpy(w2c_selected)[0:3, 0:3].to(torch.float32).to(mask_img.device)
        R = R.T
        R[:, 0:2] *= -1
        Translate = torch.from_numpy(w2c_selected)[0:3, -1].to(torch.float32).to(mask_img.device)
        Translate[0:2] *= -1

        preop_mesh_torch = Meshes(verts=[preop_mesh_hemo[:, 0:3]], faces=[faces_preop])

        cameras = PerspectiveCameras(
            focal_length=((K_base[0, 0, 0], K_base[0, 1, 1]),),
            principal_point=((K_base[0, 0, 2], K_base[0, 1, 2]),),
            R=torch.unsqueeze(R,0),
            T=torch.unsqueeze(Translate,0),
            image_size = ((img_height, img_width),),
            in_ndc=False)
        sigma_sil = 1e-5  # 1e-3 "OK" for real dataset and  1e-5 for synthetic dataset
        # ONLY for silhouette rendering
        raster_settings = RasterizationSettings(
            image_size=(img_height, img_width),  # Image size
            blur_radius=  np.log(1. / 1e-4 - 1.) * sigma_sil,
            faces_per_pixel= 28, #28
            perspective_correct=True,
        )
        sigma_depth = 0.0
        raster_settings_depth = RasterizationSettings(
            image_size=(img_height, img_width),  # Image size
            blur_radius= np.log(1. / 1e-4 - 1.) * sigma_depth,
            faces_per_pixel= 1,
            perspective_correct=True,
        )
        # ONLY for mask rasterizers(preop and Intraop)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings).to(mask_img.device)
        renderer_silhouette= MeshRenderer(rasterizer=rasterizer, shader= SoftSilhouetteShader())

        # ONLY for depth rasterizer
        rasterizer_depth = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_depth).to(mask_img.device)
        # ONLY for surface normal map rendering (preoperative model)
        renderer_normal = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_depth),
            shader=NormalShader(device=device, cameras=cameras))

        # Render the preoperative normal maps
        normal_map = renderer_normal(preop_mesh_torch)  # (1, H, W, 4)
        preop_normal = normal_map[0,..., :3]  # (H, W, 3), values in [-1,1]

        # Render Preop mask
        # Render silhouette images.  The 3rd channel of the rendering output is the alpha/silhouette channel
        silhouette_rasterized_preop = renderer_silhouette(preop_mesh_torch, cameras=cameras)[0, :, :, 3:4]  # Render Preoperative mask(silhouette)

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # silhuet_np = (preop_mask_rasterized_0).cpu().detach().numpy()
        # plt.imshow(silhuet_np)
        # plt.axis("off")
        # plt.show()


        ################################  total silhouette  ############################
        # mask_img = torch.where(mask_img > 0.0, 1.0, 0.0) # get Binary mask 0.5
        # mask_img = (mask_img > 0.0).float()
        # silhouette_rasterized_preop = torch.where(silhouette_rasterized_preop > 0.5 , 1.0, silhouette_rasterized_preop) # 0.75  get Binary preop mask
        # preop_mask_rasterized_0 = torch.where(silhouette_rasterized_preop < 0.5 , 0.0, silhouette_rasterized_preop)
        # preop_mask_rasterized_0 = silhouette_rasterized_preop


        # # Apply a strong erode to get core of the object on the rasterized silhouette
        core = self.soft_erode(x=silhouette_rasterized_preop, k=21) # k= 21
        # Apply dilation to the core (one pixel for each iteration)
        preop_mask_rasterized_0 = self.soft_dilation(
            mask=silhouette_rasterized_preop,
            core=core,
            iters=50) # iters = 50
        # preop_mask_rasterized_0 = silhouette_rasterized_preop

        ########################################  rasterize depth  #############################
        fragments = rasterizer_depth(preop_mesh_torch)
        depth_rasterized_preop = fragments.zbuf[0, ..., 0:1] # zbuffer depth for preop mesh (N,H,W)
        # Put 0 to the background
        depth_rasterized_preop = torch.where(depth_rasterized_preop <= -1, 0.0, depth_rasterized_preop)
        # depth_rasterized_preop = torch.where(preop_mask_rasterized_0 > 0.0, depth_rasterized_preop, 0.0)
########################################################################################
        #Apply mask
        if nerfacto_depth.shape[0] != mask_img.shape[0] and mask_img is not None:
            nerfacto_depth = self.resize(nerfacto_depth, scale_depth)
        if nerfacto_normal.shape[0] != mask_img.shape[0] and mask_img is not None:
            nerfacto_normal = self.resize(nerfacto_normal, scale_normal)
            # preop_normal1 = self.resize(preop_normal, scale_normal)
        nerfacto_depth_masked = nerfacto_depth * mask_img
###########################################  surface normal loss function ##############################
        # Identify smooth loss for nerfacto_normal
        # self.nerfacto_normal_smoo_loss = self.smoo_normal(nerfacto_normal)
        nerfacto_normal = F.normalize(nerfacto_normal, dim=-1)
        preop_normal_norm = F.normalize(preop_normal , dim=-1)

        # loss function from nerfacto similarity
        # dot = torch.sum(nerfacto_normal * preop_normal, dim=-1).clamp(-1, 1)
        # self.loss_normals = (mask_img[:,:,0] * (1.0 - dot)).sum() / (mask_img[:,:,0].sum() + 1e-8)   # Average only over masked region

        # Prepare every thing to plot and save them (back to [0, 1]) or shader normals
        nerfacto_normal_plot = ((nerfacto_normal + 1.0) / 2.0).clamp(0, 1) * mask_img # in [0, 1]
        preop_normal_plot = ((preop_normal_norm + 1.0) / 2.0).clamp(0, 1) # in [0, 1]

        # Apply the circle black mask (if existed)
        if os.path.exists(str(dir / "circle_mask_8")) and do_render==False:
            n = os.path.splitext(train_mask_name)[0]
            number_img = "circle_" + n.split("_")[-1][2:5] + ".png"  # choose only last four numbers
            circle_mask = torch.tensor(cv.imread(str(dir / "circle_mask_8" / number_img)), dtype=torch.bool).to(self.device)
            preop_mask_rasterized_0 = preop_mask_rasterized_0 * circle_mask[:, :, 0:1]  # Apply to rasterized mask
            # depth_rasterized_preop = depth_rasterized_preop * circle_mask[:, :, 0:1] # Apply to rasterized depth

        if step % 100 == 0 or step==max_iterations-1 and do_render == False:
            dir_meshes = os.path.join(os.path.dirname(dir), "transformed_meshes")
            save_obj(f"{dir_meshes}/mesh_step_{step:03d}.obj", preop_mesh_hemo[:, :3], faces_preop)
            np.savetxt(f"{dir_meshes}/T{step:03d}.txt", T.detach().cpu().numpy())
            depth_total = np.concatenate((mask_img[:,:,0].cpu().detach().numpy(), nerfacto_depth_masked[:,:,0].cpu().detach().numpy(), depth_rasterized_preop[:,:,0].cpu().detach().numpy(), preop_mask_rasterized_0[:,:,0].cpu().detach().numpy()), axis=1)
            plt.imsave(f"{dir_meshes}/depth_silhouette_{step:03d}.png", depth_total)
            # Plot surface normals
            normal_total = np.concatenate((nerfacto_normal_plot.cpu().detach().numpy(), preop_normal_plot.cpu().detach().numpy()), axis=1)
            plt.imsave(f"{dir_meshes}/Normals_{step:03d}.png", normal_total)

            plt.figure(figsize=(3, 3))
            # plt.imshow(depth_rasterized_preop[:,:,0].cpu().detach().numpy(), cmap='Reds', alpha=0.5)
            plt.imshow(preop_mask_rasterized_0[:,:,0].cpu().detach().numpy(), cmap='Reds', alpha=0.5)
            plt.imshow(mask_img[:,:,0].cpu().detach().numpy(), cmap='Blues', alpha=0.5)
            # red_patch = mpatches.Patch(color='red', label='Rasterized depth (Preoperative)')
            red_patch = mpatches.Patch(color='red', label='Rasterized sil (Preoperative)')
            blue_patch = mpatches.Patch(color='blue', label='Binary mask')
            # Add legend
            plt.legend(handles=[blue_patch, red_patch], loc="upper right", fontsize=3, handleheight=0.5)
            plt.axis('off')
            # Save the plot
            plt.savefig(f"{dir_meshes}/overlap_masks_{step:03d}.png", bbox_inches='tight', dpi=300)
            plt.close()

        if do_render:
            return nerfacto_depth_masked, depth_rasterized_preop, preop_mask_rasterized_0, mask_img, preop_normal
        else:
            return nerfacto_depth_masked, depth_rasterized_preop, preop_mask_rasterized_0, mask_img

    def compute_unique_edge_lengths(self, vertices, faces):
        """
        Computes lengths of unique edges in the mesh.

        Args:
            vertices: Tensor of shape (V, 3)
            faces: Tensor of shape (F, 3)

        Returns:
            lengths: Tensor of shape (E,) where E is number of unique edges
        """
        # 1. Create a list of all edges (F * 3 pairs)
        # Pair v0-v1, v1-v2, v2-v0
        edges = torch.cat([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ], dim=0)

        # 2. Sort indices within each edge so (0,1) and (1,0) are treated as the same edge
        edges, _ = torch.sort(edges, dim=1)

        # 3. Remove duplicate edges
        unique_edges = torch.unique(edges, dim=0)

        # 4. Compute lengths
        p1 = vertices[unique_edges[:, 0]]
        p2 = vertices[unique_edges[:, 1]]

        lengths = torch.norm(p1 - p2, p=2, dim=1)

        return lengths

    def compute_face_angle_cosines(self, vertices, faces, eps=1e-8):
        """
        Computes the cosine of the angles at the three corners of each face.

        Args:
            vertices: Tensor of shape (V, 3)
            faces: Tensor of shape (F, 3)
            eps: Small epsilon for numerical stability during normalization

        Returns:
            cosines: Tensor of shape (F, 3)
                     col 0: cos of angle at faces[:, 0]
                     col 1: cos of angle at faces[:, 1]
                     col 2: cos of angle at faces[:, 2]
        """
        # 1. Gather coordinates for the vertices of each face
        # Shape: (F, 3, 3) -> (num_faces, 3_vertices, xyz)
        face_verts = vertices[faces]

        # 2. Generate the three vectors originating from each corner.
        # To calculate the angle at vertex i, we need the two vectors
        # pointing away from i towards the other two vertices.

        # v_current: The vertex at which we are calculating the angle
        v_current = face_verts

        # v_next: The next vertex in the triangle (cyclic order)
        # Shift -1: 0->1, 1->2, 2->0
        v_next = torch.roll(face_verts, shifts=-1, dims=1)

        # v_prev: The previous vertex in the triangle
        # Shift +1: 0->2, 1->0, 2->1
        v_prev = torch.roll(face_verts, shifts=1, dims=1)

        # 3. Create edge vectors pointing *away* from the current vertex
        # Shape: (F, 3, 3)
        vec1 = v_next - v_current
        vec2 = v_prev - v_current

        # 4. Compute Cosine Similarity
        # dot(A, B) / (|A| * |B|)
        # dim=-1 computes along the XYZ dimension
        cosines = F.cosine_similarity(vec1, vec2, dim=-1, eps=eps)

        # Result shape is (F, 3)
        return cosines


    def loss_func(self, step, T, dir, verts, faces, rendered_depth, cam_idx, mask_img, max_iterations, pred_normal):

        nerfacto_depth, rasterized_depth, rasterized_silhouette, sil_mask = self.raster(step, T, verts, faces, dir, rendered_depth, cam_idx, mask_img, max_iterations, pred_normal, False)

        # mask_old = torch.abs(rasterized_silhouette - sil_mask).mean()
        # Adrien_depth_L1_N = torch.abs(rasterized_depth - nerfacto_depth).mean()


        # Put 0.0 on bottom part of the mask image and preop mask
        # sil_mask_half = sil_mask.clone()
        # sil_mask_half[round(sil_mask_half.shape[0]/2):,:,:] = 0.0
        # preop_mask_half = rasterized_silhouette.clone()
        # preop_mask_half[round(preop_mask_half.shape[0]/2):,:,:] = 0.0

        image_mask_bool = sil_mask > 0
        preop_mask_bool = rasterized_silhouette > 0

        # Adrien_mask_L1_N = (image_mask_bool * torch.abs(rasterized_silhouette - sil_mask)).sum() / torch.count_nonzero(image_mask_bool)
        # Adrien_mask_L1_N = (image_mask_bool * torch.abs(rasterized_silhouette - sil_mask)).sum() / (image_mask_bool.sum() + 1e-6)

        Adrien_mask_L2_N = ((rasterized_silhouette - sil_mask)**2).mean()

        # mask_old = Adrien_mask_L2_N

        # top_mask = torch.abs(preop_mask_half - sil_mask_half).mean()

        # import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # silhuet_np = (nerfacto_depth).cpu().detach().numpy()
        # # silhuet_np = (silhouette_rasterized_preop).cpu().detach().numpy()
        # plt.imshow(silhuet_np)
        # plt.axis("off")
        # plt.show()

        # Adrien_depth_L1_N = ((image_mask_bool * preop_mask_bool) * torch.abs(rasterized_depth - nerfacto_depth)).sum() / torch.count_nonzero(image_mask_bool * preop_mask_bool)
        Adrien_depth_L1_N = ((image_mask_bool * preop_mask_bool) * torch.abs(rasterized_depth - nerfacto_depth)).sum() / ((image_mask_bool * preop_mask_bool).sum() + 1e-6)


        # Intersection over Union metric (IOU)
        # intersection = (rasterized_silhouette * sil_mask).sum()
        # union = rasterized_silhouette.sum() + sil_mask.sum() - intersection
        # iou = 1 - (intersection / union)

################################ shape-preserving regularization terms ########################################
        # I = torch.eye(3, device=T.device)
        # scale = torch.abs(torch.det(T[:3,:3])) ** (1/3)
        # R_pure = T[:3,:3] / scale
        # L_ortho = torch.norm(R_pure.T @ R_pure - I) ** 2 # Orthogonality loss
        # L_det = torch.norm(torch.det(R_pure) - 1.0) ** 2  # scale loss

        # scale = torch.abs(torch.det(self.T_inv[:3,:3])) ** (1/3)
        # U, scale, V = torch.linalg.svd(T[:3,:3])  # S = singular values = scale factors
        # R_pure = T[:3,:3] / scale
        # L_ortho = torch.norm(R_pure.T @ R_pure - I) ** 2 # Orthogonality loss
        # _, scale_pure, _ = torch.linalg.svd(R_pure)
        # L_scale = torch.norm(scale_pure - 1) ** 2
        # L_scale = ((scale-pure - 1.0)**2).sum()

        depth_w = 1e2 # 1e2 Fixed Ok
        mask_w = 1e3
        # if T.shape[1] == 1:  # [N,1]  Basis deformation
        #     mask_w = 1e3 # 1e3 OK
        # else: mask_w = 1e2 # for Rigid and Affine transformations 1e2
        # put the min weight(1000) for angle loss term to prevent much deformation (ARAP)
        if T.shape[0] == 4:  # [4,4] Rigid transformation
            angle_w = 0.0
        else:
            angle_w = max(1000, 1e4 * (10 / 1000) ** (step / 100000))  # OK 1e4
        # save to dict
        # if step % 1 == 0 or step == max_iterations - 1:
            # append a dict for this step
            # self.loss_log_registration.append({
            #     "step": step,
            #     "depth_depth loss": float(Adrien_depth_L1_N.item() * depth_w),
            #     "total_loss": float(mask_old.item() * mask_w + Adrien_depth_L1_N.item() * depth_w + self.loss_angle.item() * angle_w ),
            #     "angle_loss": self.loss_angle.item(),
            #     "IOU": float(iou.item()),
            #     "T": T.tolist(),
            #     "normal": self.loss_normals.item(),
            #     "mask_old": mask_old.item() * mask_w,
            # })
        # save at the end of training
        # if step % 1000 == 0 or step == max_iterations - 1:
        #     with open(os.path.join(os.path.dirname(str(dir)), "transformed_meshes/xloss_registration.json"), "w") as f:
        #         json.dump(self.loss_log_registration, f, indent=2)


        return  Adrien_mask_L2_N * mask_w + Adrien_depth_L1_N * depth_w + self.loss_angle * angle_w


    def open_json_plot(path): # path = "/home/geintra/MIS/data/liver/liver_real/p1/transformed_meshes/xloss_registration.json" # p1
                              # path = "/home/geintra/MIS/data/liver/liver_real/20180905/transformed_meshes/xloss_registration.json" # p2
                              # path = "/home/geintra/MIS/data/liver/liver_real/P3/transformed_meshes/xloss_registration.json"  # p3
                              # path = "/home/geintra/MIS/data/liver/liver_real/P5/transformed_meshes/xloss_registration.json"  # p5
                              # path = "/home/geintra/MIS/data/nasel/P42/transformed_meshes/xloss_registration.json"  # nose
                              # path = "/home/geintra/MIS/data/uterus/transformed_meshes/xloss_registration.json"  # Uterus
                              # path = "/home/geintra/MIS/data/liver/liver_yesfall/transformed_meshes/xloss_registration.json" # old_synth
                              # path = "/home/geintra/MIS/data/liver/new_rigid/transformed_meshes/xloss_registration.json" # new_synth
        import json
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        with open(path, "r") as f:
            loss_log = json.load(f)

        loss_log = pd.DataFrame(loss_log)
        if path =="/home/geintra/MIS/data/liver/liver_real/p1/transformed_meshes/xloss_nerf.json":
            plott(loss_log["step"], loss_log["rgb"], "rgb Error", 'Iterations', "Error", "Evaluation rgb loss term")
            plott(loss_log["step"], loss_log["smoother_depth"], "smoother depth Error", 'Iterations', "Error", "Evaluation smoothness term")
        else:
            # Compute scale for each depth error
            T = loss_log["T"]
            sam = []
            for i in range(0, len(T)):
                TT = np.array(T[i])
                if TT.shape[1] == 1:
                    scale = 1
                else:
                    scale = np.abs(np.linalg.det(TT[:3, :3])) ** (1 / 3)
                T_scaled = loss_log["depth_depth loss"][i] * scale
                sam.append({"scale": scale, "depth_depth_scaled": T_scaled})
            sam_add = pd.DataFrame(sam)
            total = loss_log.join(sam_add)
            # plott(total["step"], total["silhouette_mask loss"], "Mask loss Error", 'Iterations', "Mask error","Normalized Mask loss term")
            plott(total["step"], total["depth_depth loss"], "Depth error", 'Iterations', "Depth Loss","Normalized Depth loss term")
            plott(total["step"], total["IOU"], "IOU", 'Iterations', "IOU", "Intersection Over Union")
            plott(total["step"], total["scale"], "scale", 'Iterations', "Scale factor","Rigid Transformation Scale Factor")
            plott(total["step"], total["depth_depth_scaled"], "Scaled Depth error", 'Iterations', " Depth Loss (mm)","Normalized Depth Error (mm)")
            plott(total["step"], total["total_loss"], "total_loss", 'Iterations', "total_loss loss","total_loss loss term")
            plott(total["step"], total["normal"], "Normal surface loss", 'Iterations', "Normal loss","Normal loss term")
            plott(total["step"], total["angle_loss"], "Angle loss", 'Iterations', "Angle loss","Angle loss term")
            plott(total["step"], total["mask_old"], "mask old", 'Iterations', "mask_old","mask_old")

        return()


    def plott(x, y, label:str, xlabel,ylabel,title:str):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=label, color='blue')
        # Labeling
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{label}.png", dpi=300, bbox_inches='tight')
        return()

    def get_loss_dict(self, step, outputs, batch, metrics_dict, file_dir= None, depth_img=None, cam_idx= None, mask = None, max_iterations= None, pred_normal= None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb) + self.loss_depth
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            # if step % 1 == 0 or step == max_iterations-1:
            #     # append a dict for this step
            #     self.loss_log_nerf.append({
            #         "step": step,
            #         "rgb": float(self.rgb_loss(gt_rgb, pred_rgb)),
            #         "smoother_depth": float(self.loss_depth/1e-6),})

            if step >= 1000:
                loss_dict["registration"]=self.loss_func(step, self.optim(), file_dir, self.verts,self.faces.verts_idx, depth_img, cam_idx, mask, max_iterations, pred_normal)
                #loss_dict["normal_loss"] = self.loss_normals * 1e-5  # 1e-5
            # Save the nerf evaluation parameters during training
            # if step % 1000 == 0 or step == max_iterations-1:
            #     with open(os.path.join(os.path.dirname(str(file_dir)),"transformed_meshes/xloss_nerf.json"), "w") as f:
            #         json.dump(self.loss_log_nerf, f, indent=2)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], dir,
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        # depth = outputs["depth"] * batch["mask"]
        # depth_np = depth.cpu().numpy()
        # plt.imshow(depth_np[:,:,0])
        # plt.show()

        mask_dir = dir/"mask_val_ERODE"
        specular_mask = sorted(os.listdir(mask_dir))[batch["image_idx"]]
        specular_read = torchvision.io.read_image(str(mask_dir/specular_mask)).permute([1,2,0]).bool()
        if "mask" in batch: #This mask comes from .json file
            mask = torch.cat((batch["mask"], batch["mask"], batch["mask"]), dim=2)
            # acc = torch.where(mask == 0.0, 1.0, acc)
            # acc = acc * mask
            # depth = depth * mask
            predicted_rgb = torch.where(mask == 0.0, 1.0, predicted_rgb)
            gt_rgb = torch.where(mask == 0.0, 1.0, gt_rgb)
        # Get corresponding Gt depth from "depth_2" folder
        # if os.path.exists(dir/"depths_2") and "mask" in batch:
            # depth_path = dir/"depths_2"
            # depth_GT = torchvision.io.read_image(str(depth_path / specular_mask)).permute([1, 2, 0]).cuda(0)[:, :, 0:1]
            # depth_GT = self.normalize(depth_GT)
            # depth_GT_mask = (depth_GT * mask)[:, :, 0:1]
            #get depths without highlight indices
            # depth_GT_mask_wo_spe = depth_GT_mask[specular_read[:, :, 0:1]].to(torch.float32)
            # depth_wo_spe = depth[specular_read[:, :, 0:1]]

            #Compute alpha and disparity between GT depth and rendered depth
            # depth_resh = torch.reshape(depth, [-1])
            # depth_GT_resh = torch.reshape(depth_GT_mask, [-1]).to(torch.float32)
            # alpha = torch.tensordot(torch.transpose(depth_resh, -1, 0), depth_GT_resh, dims=1) / torch.tensordot(torch.transpose(depth_resh,-1,0),depth_resh,dims=1)
            # alpha_wo_spe = torch.tensordot(torch.transpose(depth_wo_spe, -1,0), depth_GT_mask_wo_spe, dims=1) / torch.tensordot(torch.transpose(depth_wo_spe,-1,0),depth_wo_spe,dims=1)
            # disparity = torch.mean(torch.abs(depth_GT_mask - depth * alpha), dim=2)
            #Compute depth rmse
            # rmse_depth_wo_bg = self.rmse(depth_GT_resh, depth_resh * alpha) # Error for wo bg
            # rmse_depth_wo_spe = self.rmse(depth_GT_mask_wo_spe, depth_wo_spe * alpha_wo_spe)  #Error for wo spe


            #display disparity and convert it into torch to save them
            # fig = self.plot_disprity(disparity, rmse_depth_wo_bg, rmse_depth_wo_spe)
            # buf = io.BytesIO()
            # fig.savefig(buf, format='png')
            # buf.seek(0)
            # Read the buffer into a PIL image
            # image = Image.open(buf)
            # Convert the PIL image to a PyTorch tensor
            # tensor_disparity = torch.tensor(np.array(image)) / 255.0  # Normalize to [0, 1]
            # combined_disparity = torch.cat([tensor_disparity], dim=1)
            # combined_depth_GT = torch.cat([depth_GT_mask.to(torch.float32)], dim=1)
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Calculate psnr without specular pixels
        predicted_rgb_wo_specular = predicted_rgb[specular_read]
        gt_rgb_wo_specular = gt_rgb[specular_read]
        psnr_wo_spe = self.psnr(gt_rgb_wo_specular, predicted_rgb_wo_specular)
        # Calculate psnr, ssim and lpips with bg 0 (masked) if available
        if "mask" in batch:
            gt_rgb_w_bg_0 = torch.moveaxis(gt_rgb * mask, -1, 0)[None, ...]
            predicted_rgb_w_bg_0 = torch.moveaxis(predicted_rgb * mask, -1, 0)[None, ...]
            psnr_wo_bg = self.psnr(gt_rgb[mask], predicted_rgb[mask])
            ssim_w_bg_0 = self.ssim(gt_rgb_w_bg_0, predicted_rgb_w_bg_0)
            lpips_w_bg_0 = self.lpips(gt_rgb_w_bg_0, predicted_rgb_w_bg_0)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        # metrics_dict = {"psnr_wo_spe": float(psnr_wo_spe.item())}
        metrics_dict["psnr_wo_spe"] = float(psnr_wo_spe)
        if "mask" in batch:
            metrics_dict["psnr_wo_bg"] = float(psnr_wo_bg)
            metrics_dict["ssim_w_bg_0"] = float(ssim_w_bg_0)
            metrics_dict["lpips_w_bg_0"] = float(lpips_w_bg_0)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        # if os.path.exists(dir / "depths_2") and "mask" in batch:
            # Compute rmse depth between masked GT depth and masked rendered depth map
            # metrics_dict["rmse_depth_wo_bg"] = float(rmse_depth_wo_bg)
            # metrics_dict["rmse_depth_wo_spe"] = float(rmse_depth_wo_spe)
            # images_dict["disparity"] = combined_disparity # Add disparity map into image_dict
            # images_dict["depth_GT"] = combined_depth_GT


        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            if "mask" in batch:
                prop_depth_i = prop_depth_i
            else:
                prop_depth_i = prop_depth_i
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def plot_disprity(self, disparity, rmse_value_bg, rmse_value_spe):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        rmse_bg = float(rmse_value_bg)
        rmse_spe = float(rmse_value_spe)
        frame_error = "Depth RMSE_wo_bg: "+ str(round(rmse_bg, ndigits= 4))+"(Mm)"+"Depth RMSE_wo_spe: "+ str(round(rmse_spe, ndigits=4))+"(Mn)"
        disparity = self.normalize(disparity) # Normalization
        # Plot dynamic colormap for disparity map
        fig = plt.figure()
        ax = fig.add_subplot()
        # # I like to position my colorbars this way, but you don't have to
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        cv0 = torch.Tensor.cpu(disparity).numpy()
        im = ax.imshow(cv0)  # Here make an AxesImage rather than contour
        # cb = fig.colorbar(im, cax=cax)
        # tx = ax.set_title(framename)
        # plt.savefig(output_path)
        fig.colorbar(im, cax=cax)
        ax.set_title(frame_error)
        return(fig)

    def sobel_contour(self,image: torch.Tensor) -> torch.Tensor:
        """
        Compute edge (contour) map of a silhouette using Sobel operator.

        Args:
            image: Tensor of shape (B, 1, H, W), values in [0, 1], silhouette mask.

        Returns:
            edge_map: Tensor of shape (B, 1, H, W), gradient magnitude (edge strength).
        """
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)

        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # avoid sqrt(0)
        return edge_map

    def normalize(self, x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 10e-10)
    def rmse(self, actual, pred): return torch.sqrt(torch.square(torch.subtract(actual, pred)).mean())


    def gaussian_blur(self, input, kernel_size=3, sigma=0.8):
        """
        Apply a 2D Gaussian blur on a grayscale or batched image tensor.

        Args:
            input: Tensor of shape (B, 1, H, W) or (1, H, W)
            kernel_size: Size of the Gaussian kernel (odd number)
            sigma: Standard deviation of the Gaussian kernel

        Returns:
            Blurred image of the same shape
        """
        if input.dim() == 3:  # Convert (1, H, W) to (1, 1, H, W)
            input = input.unsqueeze(0)
        B, C, H, W = input.shape

        # Create 1D Gaussian kernel
        ax = torch.arange(kernel_size, dtype=torch.float32, device=input.device) - kernel_size // 2
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss = gauss / gauss.sum()

        # Make 2D Gaussian kernel by outer product
        kernel_2d = torch.outer(gauss, gauss)
        kernel_2d = kernel_2d.expand(C, 1, kernel_size, kernel_size)  # (C,1,k,k)

        # Apply depthwise conv2d for each channel
        blurred = F.conv2d(input, kernel_2d, padding=kernel_size // 2, groups=C)
        blurred = torch.squeeze(blurred,0)

        return blurred

    def resize(self, tensor: torch.Tensor, scale_factor) -> torch.Tensor:
        '''

        :param tensor: The tensor which should be resized.
        :return: resized tensor.
        '''
        if tensor.shape[-1] == 1:
            tensor = tensor * 1
            tensor = torch.cat((tensor, tensor, tensor), dim=-1)
        image = tensor.unsqueeze(0)
        image = torch.permute(image, dims=(0, 3, 1, 2))
        image = image.float()
        resized_image = F.interpolate(image, scale_factor=1 / scale_factor, mode='bilinear', align_corners=False)
        resized_image = resized_image.squeeze(0)
        return torch.permute(resized_image, dims=(1, 2, 0))

# Custom shader: directly return normals
class NormalShader(SoftPhongShader):
    def forward(self, fragments, meshes, **kwargs):
        # Interpolate vertex normals for each pixel
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces = meshes.faces_packed()  # (F, 3)
        face_normals = vertex_normals[faces]  # (F, 3, 3)

        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_normals
        )  # (N, H, W, K, 3)
        # Just return normals in [-1,1], with alpha=1
        normal_map = torch.cat([pixel_normals[..., 0, :], torch.ones_like(pixel_normals[..., 0, :1])], dim=-1)
        return normal_map

