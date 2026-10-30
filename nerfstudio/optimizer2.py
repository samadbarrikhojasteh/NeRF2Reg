import pytorch3d.io
from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings, MeshRasterizer,MeshRenderer,
    MeshRasterizer, SoftPhongShader,SoftSilhouetteShader,SoftPhongShader,TexturesVertex,PointLights)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import json
import cv2
import os
import pytorch3d.transforms as transforms
from pytorch3d.io import load_obj , save_obj
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pytorch3d.io import load_obj , save_obj, load_ply, save_ply
import torch

# Function to build a transformation matrix from rotation and translation parameters
def build_transformation_matrix(rotation_params, translation, scale, device):
    # Compute rotation matrix from Euler angles
    R = transforms.euler_angles_to_matrix(rotation_params,'ZYX').to(device)
    # q = transforms.matrix_to_quaternion(R)
    # q_norm = rotation_params / (rotation_params.norm(dim=-1, keepdim=True) + 1e-7) # normalize the quaternion
    # R = transforms.quaternion_to_matrix(q_norm)
    # R = transforms.rotation_6d_to_matrix(rotation_params)
    # Apply the scale to the rotation component only
    R_scale = R * scale
    # Create a 4x4 transformation matrix
    T = torch.eye(4, device=device)
    T[:3, :3] = R_scale
    T[:3, 3] = translation
    return T

# theta = torch.tensor([[ 0.14409302,  0.10641164, -0.03093669],
#               [ 0.07633032, -0.05851513,  0.15424972],
#               [ 0.08033803, -0.13526218, -0.09106734]])
# Convert procrustes's rotation matrix to the radiance(angle)
def rot2eul(theta):
    scale = torch.linalg.norm(theta, axis=0)
    eul = transforms.matrix_to_euler_angles(theta/scale, "ZYX")
    return eul, scale[0]
# eul, scale = rot2eul(R)
###################################################################
def get_preop(dir):
     verts_preop, faces_preop, properties_preop = load_obj(dir,load_textures=True, create_texture_atlas=True, device=device)
     return verts_preop, faces_preop

# verts, faces = get_preop("/home/geintra/MIS/data/liver/liver_real/p1/models/liver.obj")  # P1
#  verts, faces = get_preop("/home/geintra/MIS/data/liver/liver_real/20180905/data/models/liver.obj") # P2
# # Compute center of the mass
# center = torch.mean(verts, dim=0, keepdim=True)
# mesh_center = (verts-center)*(0.03)
# mesh_center[:,0:2]*=-1
# # Rotation: 90° clockwise around Z
# angle = -75 * torch.pi / 180  # radians
# axis = torch.tensor([0, 0, 1])  # Z-axis
# R = (transform.axis_angle_to_matrix(angle * axis[None, :])).to(device)  # (1, 3, 3)
# eul = transform.matrix_to_euler_angles(R, "ZXY")
# # Apply to your mesh vertices (N, 3)
# vertices_rotated = torch.matmul(mesh_center, R.squeeze(0).T)+torch.tensor([-5.0, 0.0, 0.0]).to(device)
#
# save_obj("mesh_center.obj", vertices_rotated, faces.verts_idx)
#########################################################

# Define the differentiable function for optimizing the transformation
class MeshAligner(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize rotation angles (in radians), translation, and scale parameters

        self.rotation_params =   nn.Parameter(torch.tensor([0.0, 0.0, 0.0], requires_grad= True, device=device))
        self.translation =  nn.Parameter(torch.tensor([0.0, 0.0, 0.0], requires_grad= True, device=device))
        self.scale =   nn.Parameter(torch.tensor([1.0], requires_grad= True, device=device))

    def forward(self):
        # Get the transformation matrix
        T = build_transformation_matrix(self.rotation_params, self.translation, self.scale, device)
        return T


class Basis(nn.Module):
    def __init__(self):
        super().__init__()
        basis_load = np.load("/home/geintra/MIS/basis_10_p1.npy")  # P1
        basis = torch.from_numpy(basis_load).to(torch.float32)
        self.basis_nn = nn.Parameter(basis)
    def forward(self):
        return self.basis_nn


class Deformation(nn.Module):
    def __init__(self):
        super().__init__()
        deform_matrix = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
        self.deform_matrix = nn.Parameter(deform_matrix)
    def forward(self):
        return self.deform_matrix

class Spline(nn.Module):
    def __init__(self):
        super().__init__()
        self.spline_matrix = nn.Parameter(torch.tensor(torch.zeros(48,1), device=device)) # 12 , 24 , 48 (for 4 , 8 and 16 basis)
    def forward(self):
        return self.spline_matrix


class Affine(nn.Module):
    def __init__(self):
        super().__init__()

        self.affine_matrix = nn.Parameter(torch.tensor(torch.eye(4)[:3,:], device=device)) # P1 "OK"

        # self.affine_matrix = nn.Parameter(torch.tensor([[-16.5038,   24.4214,   -16.5947, -85.5767],
        #                                                       [ -29.0778,  -16.7268,   4.3138,  -138.7156],
        #                                                       [ -5.0940,   16.3717,   29.1604,  -40.3736]], device=device)) # P1 "OK"
        # self.affine_matrix = nn.Parameter(torch.tensor([[  9.6585,   29.6820,  -16.6417,  -10.8585],
        #                                                       [ -32.0391,    2.1034,  -14.8434, -197.6046],
        #                                                       [ -11.4658,   19.1261,   27.4588,  -77.8513]], device=device)) # P2 "OK" (20180905)

        # self.affine_matrix = nn.Parameter(torch.tensor([[ 8.8026e-02, -4.8090e-02,  1.3734e-01, -1.4460e+00],
        #                                                 [ 1.3064e-01,  9.6804e-02, -4.9839e-02, -3.0362e-02],
        #                                                 [-6.4081e-02,  1.3130e-01,  8.7045e-02, -3.3759e-01]], device=device)) # Synthetic inv

        #self.affine_matrix = nn.Parameter(torch.tensor([[ 0.14409302,  0.10641164, -0.03093669,  5.13048906],
        #                                                     [ 0.07633032, -0.05851513,  0.15424972,  2.49911774],
        #                                                     [ 0.08033803, -0.13526218, -0.09106734,  6.37960113]], device=device)) # Synthetic_W/O_prob (preop vs img)
        # self.affine_matrix  = nn.Parameter(torch.tensor([[ 3.0434,  4.5170, -2.2156,  3.7900],
        #                                                        [-1.6626,  3.3469,  4.5396, -0.7700],
        #                                                        [ 4.7484, -1.7232,  3.0096,  7.8300]], device=device)) # syn liver OK

        # self.affine_matrix = nn.Parameter(torch.tensor([[ -20.47043412,   23.00819992,  -23.83985825, -141.60671631],
        #                                                      [ -26.14875223,  -28.42819896,   -4.98347658, -294.63033385],
        #                                                      [ -20.34597901,   13.38712137,   30.3904738 ,  -90.33272331]], device=device))  # for P5
        #
        # self.affine_matrix = nn.Parameter(torch.tensor([[ 7.3661,  52.4739,   2.0896,  24.2069],
        #                                                      [-51.7105,   7.6155,  -8.9535, -87.3690],
        #                                                      [ -9.1598,  -0.7939,  52.2265, -68.7959]], device=device)) # Uterus
    def forward(self):
        return self.affine_matrix



# path = "/home/geintra/MIS/data/liver/liver_real/p1/models/liver.obj" # P1
# path_half = "/home/geintra/MIS/data/liver/liver_real/p1/models/liver_half.obj" # half P1
# path_colmap = "/home/geintra/MIS/data/liver/liver_real/p1/models/mesh_mod.ply" # Colmap reconstruction
########
# path = "/home/geintra/MIS/data/liver/liver_real/20180905/data/models/liver.obj" # P2
# path_half = "/home/geintra/MIS/data/liver/liver_real/20180905/data/models/half_liver.obj" # Half P2
# path_colmap = "/home/geintra/MIS/data/liver/liver_real/20180905/data/models/mesh_colmap1.ply" # Colmap reconstruction
###########
# path = "/home/geintra/MIS/data/liver/liver_real/P5/models/liver.obj" # P5
# path_half = "/home/geintra/MIS/data/liver/liver_real/P5/models/half_liver1.obj" # Half P5
# path_colmap = "/home/geintra/MIS/data/liver/liver_real/P5/models/colmap1.ply"    # Colmap reconstruction
############
# path = "/home/geintra/MIS/data/uterus/models/uterus2.obj" # Uterus
# path_half = "/home/geintra/MIS/data/uterus/models/uterus2_half1.obj" # Half Uterus
# path_colmap = "/home/geintra/MIS/data/uterus/models/point_colmap.ply" # Colmap reconstruction
#############
# path = "/home/geintra/MIS/data/nasel/P42/models/updated_2.obj" # Nose
# path = "/home/geintra/MIS/data/nasel/P42/models/Maxillary ostium LEFT.obj" # Maxillary Nose
# path = "/home/geintra/MIS/data/liver/liver_yesfall/output_total/GT_preop_non_rigid.obj" # synthetic non-rigid(modified)
# path = "/home/geintra/MIS/data/liver/liver_yesfall/output_total/gt_mesh/rigid/preop/higado_simple.obj" # synthetic org (rigid)
# #############
path = ("/home/geintra/MIS/data/liver/new_rigid/models/synth_new.obj") # Synthetic_new
path_half = ("/home/geintra/MIS/data/liver/new_rigid/models/synth_new_half.obj") # half Synthetic_new
path_colmap = ("/home/geintra/MIS/data/liver/new_rigid/models/mesh_colmap1.ply") # colmap Synthetic_new
###############
# path = ("/home/geintra/MIS/data/liver/deformation_NR1/model/max_deform448_sim.obj") # Synthetic_NR1
# path_half = ("/home/geintra/MIS/data/liver/deformation_NR1/model/max_deform448_sim1.obj") # half Synthetic_NR1
# path_colmap = ("/home/geintra/MIS/data/liver/deformation_NR1/model/mesh_colmap1.ply") # colmap Synthetic_NR1
################
#
# verts_preop, faces_preop, _ = load_obj(path,load_textures=False, create_texture_atlas=True, device=device)
# verts_preop_half, faces_preop_half, _ = load_obj(path_half,load_textures=False, create_texture_atlas=True, device=device)
# # Pre initial manually for liver_half
# x_half = verts_preop_half[:, 0]
# y_half = verts_preop_half[:, 1]
# z_half = verts_preop_half[:, 2]
# # verts_preop_half_new = torch.stack([x_half,-z_half,-y_half], dim=-1)
# verts_preop_half_new = torch.stack([-z_half, -y_half, -x_half], dim=-1)
#
# # Pre initial manually for whole liver
# x = verts_preop[:, 0]
# y = verts_preop[:, 1]
# z = verts_preop[:, 2]
# verts_preop_new = torch.stack([-z, -y, -x], dim=-1) # [x,-z, y] for real and [-z, -y, -x] for synth

# # Apply degree correction into identity rotation (R) ** Not in Uterus and synthetic datasets**
# R = rotation_matrix_xyz(rx= 50, ry= 0, rz= 0).to(device  = device) # (rx= 10, ry= 0, rz= 30) for real datasets  ###
# verts_preop_half_new = verts_preop_half_new @ R.T
# verts_preop_new = verts_preop_new @ R.T
# # save_obj("half_liver_preop_half.obj",verts_preop_half_new, faces_preop_half.verts_idx)
# #
# verts_preop_colmap = load_ply(path_colmap)[0].to(device=device)

# transformation = torch.tensor([
#         [
#             0.03736924007534981,
#             0.12359344959259033,
#             -0.9916290640830994,
#             0.09131030738353729
#         ],
#         [
#             0.9984065890312195,
#             0.03736924007534981,
#             0.042282234877347946,
#             -0.05419836565852165
#         ],
#         [
#             0.042282234877347946,
#             -0.9916290640830994,
#             -0.12200009822845459,
#             0.005276462063193321
#         ]
#     ], device=device)  # synth_NR1
# scale = torch.tensor([ 0.14168848817517238], device=device) # synth_NR1
#
transformation = torch.tensor([
        [
            0.013957420364022255,
            -0.03143119812011719,
            -0.999408483505249,
            0.1319112628698349
        ],
        [
            0.9998111128807068,
            0.013957420364022255,
            0.013524086214601994,
            -0.08220010250806808
        ],
        [
            0.013524086214601994,
            -0.999408483505249,
            0.031620144844055176,
            -0.006298742722719908
        ]
    ], device=device)  # new_synth
scale = torch.tensor([ 0.17249171264947066], device=device) # new_synth
#
# ## Simulate R,t and s using PCA for pointcloud_Colmap and Preoperative
# ## Transfer Preoperative coordinate to COLMAP world coordinate
# R_idt, t, s = estimate_similarity(verts_preop_half_new, verts_preop_colmap)
# #Apply degree correction into identity rotation (R)
# aligned_points = transform_points(verts_preop_new, R_idt, t, s)
# save_obj("preop_to_sfm.obj", aligned_points[:, 0:3], faces_preop.verts_idx)
# #### Transfer the COLMAP world coordinate to NeRF world coordinate
# T_scale = transformation * scale
# T = torch.eye(4, device=device)
# T[:3, :] = T_scale
# #### ones = torch.ones((verts_preop.shape[0], 1), device=verts_preop.device)
# ones = torch.ones((aligned_points.shape[0], 1), device=verts_preop.device)
# homogeneous_preop = torch.cat([aligned_points, ones], dim=1).to(device=device)
# M = homogeneous_preop @ torch.transpose(T, 0, 1)
# # save_ply("preop_ply2.ply", M[:, 0:3])
# save_obj("preop_to_NeRF.obj", M[:, 0:3], faces_preop.verts_idx)




def rotation_matrix_xyz(rx, ry, rz, degrees=True):
    """
    rx, ry, rz: rotations about X, Y, Z
    Returns:
        R: (3,3) rotation matrix
    """
    import math
    if degrees:
        rx = math.radians(rx)
        ry = math.radians(ry)
        rz = math.radians(rz)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=torch.float32)

    Ry = torch.tensor([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=torch.float32)

    Rz = torch.tensor([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,   0,  1]
    ], dtype=torch.float32)

    # Apply X, then Y, then Z
    R = Rz @ Ry @ Rx
    return R

def estimate_similarity(preop_points, colmap_points):
    """
    Estimate similarity transform using PCA.

    Parameters
    ----------
    preop_points : (N,3)
    colmap_points : (M,3)

    Returns
    -------
    R : (3,3)
    t : (3,)
    s : scalar
    """
    import math
    c1, V1, L1 = compute_pca(preop_points)
    c2, V2, L2 = compute_pca(colmap_points)

    # Rotation
    # R = V2 @ V1.T
    R = torch.eye(3, device=device)

    # Scale
    # spread1 = torch.sqrt(L1.sum())
    # spread2 = torch.sqrt(L2.sum())
    # s = spread2 / spread1

    preop_rot = preop_points @ R.T
    d1 = torch.norm(preop_rot - preop_rot.mean(0), dim=1)
    d2 = torch.norm(colmap_points - colmap_points.mean(0), dim=1)
    s = torch.quantile(d2, .98) / torch.quantile(d1, .98) # 0.98

    # Translation
    t = c2 - s * (R @ c1)

    return R, t, s

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def compute_pca(points):
    """
    Compute PCA of a point cloud.

    Parameters
    ----------
    points : (N,3) torch.Tensor

    Returns
    -------
    centroid : (3,)
    eigenvectors : (3,3)
    eigenvalues : (3,)
    """

    # Centroid
    centroid = points.mean(dim=0)

    # Center the cloud
    centered = points - centroid

    # Covariance matrix
    cov = centered.T @ centered / (points.shape[0] - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Make the basis right-handed
    if torch.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    return centroid, eigenvectors, eigenvalues


def transform_points(points, R, t, s):
    """
    Apply similarity transformation.
    """
    return s * (points @ R.T) + t




# Transformation matrix (transform from Colmap to NeRF coordinate) extracted by NeRF
# transformation = torch.tensor([
#         [
#             0.10558173060417175,
#             -0.009847640991210938,
#             -0.9943618774414062,
#             -0.17764461040496826
#         ],
#         [
#             0.9889612197875977,
#             0.10558173060417175,
#             0.10396265238523483,
#             -0.6134981513023376
#         ],
#         [
#             0.10396265238523483,
#             -0.9943618774414062,
#             0.02088642120361328,
#             -0.1590431183576584
#         ]
#     ], device=device)  # P2
# scale = torch.tensor([ 0.10895356773952955], device=device) # P2
#
# transformation = torch.tensor([
#         [
#             0.0015763447154313326,
#             -0.052342891693115234,
#             -0.9986279010772705,
#             -0.5325493216514587
#         ],
#         [
#             0.999997615814209,
#             0.0015763447154313326,
#             0.0014958829851821065,
#             0.3717304468154907
#         ],
#         [
#             0.0014958829851821065,
#             -0.9986279010772705,
#             0.05234527587890625,
#             0.05160505324602127
#         ]
#     ], device=device)  # P1
# scale = torch.tensor([ 0.09911630769975333], device=device) # P1
#
# transformation = torch.tensor([
#         [
#             0.007734190672636032,
#             -0.08254122734069824,
#             -0.9965576529502869,
#             0.06785258650779724
#         ],
#         [
#             0.9999447464942932,
#             0.007734190672636032,
#             0.007119883317500353,
#             -0.141529381275177
#         ],
#         [
#             0.007119883317500353,
#             -0.9965576529502869,
#             0.08259642124176025,
#             -0.2325238734483719
#         ]
#     ], device=device)  # Uterus
# scale = torch.tensor([0.13657974910744405], device=device) # uterus

# transformation = torch.tensor([
#         [
#             0.28715574741363525,
#             0.06923472881317139,
#             -0.955378532409668,
#             -0.30069321393966675
#         ],
#         [
#             0.9114079475402832,
#             0.28715574741363525,
#             0.29474934935569763,
#             -0.3971666097640991
#         ],
#         [
#             0.29474934935569763,
#             -0.955378532409668,
#             0.01935732364654541,
#             -0.017503278329968452
#         ]
#     ], device=device)  # P5
# scale = torch.tensor([0.12089798149017149], device=device) # P5
#
# transformation = torch.tensor([
#         [
#             0.013957420364022255,
#             -0.03143119812011719,
#             -0.999408483505249,
#             0.1319112628698349
#         ],
#         [
#             0.9998111128807068,
#             0.013957420364022255,
#             0.013524086214601994,
#             -0.08220010250806808
#         ],
#         [
#             0.013524086214601994,
#             -0.999408483505249,
#             0.031620144844055176,
#             -0.006298742722719908
#         ]
#     ], device=device)  # new_synth
# scale = torch.tensor([ 0.17249171264947066], device=device) # new_synth












def loss_func(T, vertices_preop, faces_preop, intraop, faces_intra):
    rendered_depth_preop, rendered_depth_intraop = raster(T, vertices_preop, faces_preop, intraop, faces_intra)
    idx = torch.where(rendered_depth_intraop > 0)
    loss = (torch.abs(rendered_depth_preop[idx] - rendered_depth_intraop[idx])).mean()
    return loss

def raster(T , preop_mesh, faces_preop, intraop, faces_intraop):
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    dir = "/home/geintra/nerfstudio/data/liver/liver_yesfall/output_total_disparity"
    # dir = "/home/geintra/nerfstudio/data/liver/deformation/output_total"

    # depth_dir = os.path.join(dir, "depth_org_size")
    # Convert points to homogeneous coordinates
    ones = torch.ones((preop_mesh.shape[0], 1), device=device)
    homogeneous_preop = torch.cat([preop_mesh, ones], dim=1)
    # Apply the transformation
    T_inv = torch.linalg.inv(T)
    preop_mesh_hemo = homogeneous_preop @ torch.transpose(T_inv, 0, 1)

    # Get K and c2w from json file
    frame_name_total, K, distortion, c2w = readjson(dir)
    # Apply NeRFacto modifications to the c2w
    w2c = nerfacto2opencv(c2w)
    K_base = torch.tensor(np.append(K, [np.array([0.0, 0.0, 0.0, 1.0])], axis=0), dtype=torch.float32, device=device)
    K_base = torch.unsqueeze(K_base, dim=0)

    frame = []
    name = []
    sam = np.stack(w2c)
    for i in range(0, len(sam)):
        frame.append(sam[i]["frame"])
        name.append(sam[i]["name"])
    frame_np = np.array(frame)

    R = torch.from_numpy(frame_np)[:, 0:3, 0:3].to(torch.float32).to(device)
    R = torch.transpose(R, 1, 2)
    R[:, :, 0:2] *= -1
    Translate = torch.from_numpy(frame_np)[:, 0:3, -1].to(torch.float32).to(device)
    Translate[:, 0:2] *= -1

    img_width, img_height = 1920, 1080
    # Construct the Meshes object with the transformed vertices, faces, and textures
    preop_mesh = Meshes(verts=[preop_mesh_hemo[:,0:3]], faces=[faces_preop.verts_idx]).to(device)
    # Construct the Meshes object with the intraop vertices, faces, and textures
    intra_mesh = Meshes(verts=[intraop[:, 0:3]], faces=[faces_intraop.verts_idx]).to(device)
    # number of camera positions
    N = 1
    cameras = PerspectiveCameras(
        focal_length=((K_base[0, 0, 0], K_base[0, 1, 1]),),
        principal_point=((K_base[0, 0, 2], K_base[0, 1, 2]),),
        R = R[0:N, :, :],
        T = Translate[0:N, :],
        image_size=((img_height, img_width),),
        in_ndc=False)

    raster_settings = RasterizationSettings(
        image_size=(img_height, img_width),  # Image size
        blur_radius=0.0,  # No blur
        faces_per_pixel=1,  # Only the closest face per pixel
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings).to(device)

    # renderer_silhouette= MeshRenderer(rasterizer=rasterizer, shader=SoftSilhouetteShader())
    # Render silhouette images.  The 3rd channel of the rendering output is
    # the alpha/silhouette channel
    # silhouette_preop_alpha = renderer_silhouette(preop_mesh, cameras=cameras, lights=lights)[:, :, :, 3:4]
    # silhouette_intra_alpha = renderer_silhouette(intra_mesh, cameras=cameras, lights=lights)[:, :, :, 3]

    # Visualization silhouette
    # silhuet_np = silhouette_preop_alpha[0,...].cpu().detach().numpy()
    # plt.imshow(silhuet_np, cmap="gray")
    # plt.axis("off")
    # plt.show()

    if N > 1:
        #depth
        fragments = rasterizer(preop_mesh.extend(N))
        fragments_intra = rasterizer(intra_mesh.extend(N))
        # silhoutte
        # silhouette_preop_alpha = renderer_silhouette(preop_mesh.extend(N), cameras=cameras)[:, :, :, 3:4]
        # silhouette_intra_alpha = renderer_silhouette(intra_mesh.extend(N), cameras=cameras)[:, :, :, 3:4]

    else:
        #depth
        fragments = rasterizer(preop_mesh)
        fragments_intra = rasterizer(intra_mesh)
        #Silhoutte
        # silhouette_preop_alpha = renderer_silhouette(preop_mesh, cameras=cameras)[:, :, :, 3:4]
        # silhouette_intra_alpha = renderer_silhouette(intra_mesh, cameras=cameras)[:, :, :, 3:4]

    depth_rendered_preop = fragments.zbuf[:, ...,0] # zbuffer depth for preop mesh (N,H,W,1)
    depth_rendered_intra = fragments_intra.zbuf[:, ..., 0] # zbuffer depth for intra mesh (N,H,W,1)
    # sil_rendered_preop = silhouette_preop_alpha[:, ..., 0]  # silhouette for the preop (N,H,W,1)
    # sil_rendered_intra = silhouette_intra_alpha[:, ..., 0] # silhouette for the intraop (N,H,W,1)

    #filter out -1 and replace with 0

    # depth_rendered_preop = torch.where(depth_rendered_preop == -1, 0 , depth_rendered_preop)
    # depth_rendered_intra = torch.where(depth_rendered_intra == -1, 0, depth_rendered_intra)

    # Read the corresponding mask
    # mask_dir = os.path.join(dir, "masks")
    # list_mask = sorted(os.listdir(mask_dir))
    # list_N = list_mask[0:N]
    # list_gt = []
    # for path in list_N:
    #     mask_0 = os.path.join(mask_dir, path)
    #     mask_gt = cv2.imread(mask_0)[:, :, 0]
    #     list_gt.append(mask_gt)
    # assert len(list_gt) == N
    # mask_gt = np.stack(list_gt)
    # # Read the depth GT
    # list_depth = sorted(os.listdir(depth_dir))
    # list_depth_N = list_depth[0:N]
    # list_depth_list = []
    # for path_depth in list_depth_N:
    #     depth_0 = os.path.join(depth_dir, path_depth)
    #     depth_gt = cv2.imread(depth_0, cv2.IMREAD_UNCHANGED)[:,:,0:1]
    #     list_depth_list.append(depth_gt)
    # assert len(list_depth_list) == N
    # depth_gt = np.stack(list_depth_list)*1.2167/0.5459
    # Put -1 as background in depth_gt
    # depth_gt_updated = np.where(mask_gt != 0, depth_gt[:,:,0], -1)
    # depth_gt_updated = depth_gt_updated[:,:,:,None]
    # depth_perm = torch.from_numpy(depth_gt_updated).to(device) #permute(0,3, 1, 2)
    # return depth_rendered, depth_perm
    return depth_rendered_preop, depth_rendered_intra


def optimize(vertices_preop, faces_preop, intraop, faces_intraop):
    # Instantiate the model and optimizer
    model = MeshAligner()
    optimizer = optim.Adam([model.translation, model.rotation_params, model.scale], lr= 0.007) # 0.0007
    # Optimization loop
    num_iterations = 100
    for i in range(num_iterations):
        optimizer.zero_grad()
        # Forward pass (compute loss)
        modified_T = model()
        loss = loss_func(modified_T, vertices_preop, faces_preop, intraop, faces_intraop)
        # Backward pass (compute gradients and update parameters)
        loss.backward()
        # print("Parameter gradients:", model.rotation_params.grad)
        optimizer.step()
        # Print loss every 10 iterations
        if i % 10 == 0:
            print(f"Iteration {i}/{num_iterations}, Loss: {loss.item()}")
        #if i> 150:
        #    optimizer.param_groups[0]['lr'] = 0.0001
    # After optimization, the optimized transformation matrix is available
    rotation = model.rotation_params
    translation = model.translation
    optimized_scale = model.scale
    transformation_matrix = build_transformation_matrix(rotation, translation, optimized_scale, device)
    return transformation_matrix

def runtool():
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    verts_preop, faces_preop, properties_preop = load_obj("/home/geintra/MIS/gt_mesh/rigid/preop/higado_simple2.obj", load_textures= True,create_texture_atlas= True, device=device)
    # T = torch.Tensor([[8.340910822153091431e-02, -4.920966923236846924e-02, 1.166737228631973267e-01, -1.238265872001647949e+00],
    #                   [1.188136339187622070e-01, 7.884516566991806030e-02, -5.169475451111793518e-02, -2.482619509100914001e-02],
    #                   [-4.392426460981369019e-02, 1.198453232645988464e-01, 8.197957277297973633e-02, -3.543672561645507812e-01],
    #                   [-2.585829328438649100e-10, 8.495009140574438788e-10, 1.740558808194236917e-09, 1.000000000000000000e+00]]).to(device)
    # ones = torch.ones((verts_preop.shape[0], 1), device=device)
    # homogeneous_preop = torch.cat([verts_preop, ones], dim=1)
    # new_preop = homogeneous_preop@T.T
    # save_obj("mesh_step.obj", new_preop[:, :3], faces_preop.verts_idx)

    verts_intra, faces_intra, properties_intra = load_obj("/home/geintra/MIS/gt_mesh/rigid/mesh.obj", load_textures= True,create_texture_atlas= True, device=device)

    dir = "/home/geintra/nerfstudio/data/liver/liver_yesfall/output_total_disparity"
    # Find the best differentiable transformation matrix
    procrusted = optimize(vertices_preop=verts_preop, faces_preop=faces_preop,intraop=verts_intra,faces_intraop=faces_intra)
    # rasterize the depth of the transformed preoperative mesh under certain camera poses
    rendered_depth_preop, rendered_depth_intra = raster(procrusted, verts_preop, faces_preop, verts_intra, faces_intra)
    N = len(rendered_depth_intra)
    # Save rasterized silhouette OR depth in folder
    dir_total = os.path.join(dir, "rasterized_depth")
    os.makedirs(dir_total, exist_ok=True)
    for i in range(0, N):
        rendered_depth_np = torch.Tensor.detach(rendered_depth_preop[i, :, :]).cpu().numpy()
        gt_intraop = torch.Tensor.detach(rendered_depth_intra[i, :, :]).cpu().numpy()
        depth_total = np.concatenate((rendered_depth_np, gt_intraop), axis=1)
        output_depth = os.path.join(dir_total, f'depth_{i:03}.png')  # Format sequence as 'image_001.png'
        plt.imsave(output_depth, depth_total)
        plt.close()
    return ()

def rotation_matrix(a, b):
    """
    Compute the rotation matrix that rotates vector a to vector b.
    :param a: The vector to rotate
    :param b: The vector to rotate to.
    :return: The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    return np.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))

def readjson(dir: str):
    """
    Read json file and extract data
    poses are extracted along corresponding image names
    Parameters:
         dir: json file path
    Returns:
        frame_name_total: Global path of each image(all dataset images).
        K: intrinsic matrix .
        distortion: Distortion parameters as a vector .
        posewithname: Dict {poses(4*4) and name of the images}.
    """
    transform = open(os.path.join(dir, "transforms.json"))
    transforms = json.load(transform)
    # transforms["frames"][0]["k1"]
    # #Get distortion parameters
    if "w" not in transforms:
        total_transforms = transforms["frames"][0]
    else:
        total_transforms = transforms


    distortion_params = dict(
        k1=total_transforms["k1"] if "k1" in total_transforms else 0.0,
        k2=total_transforms["k2"] if "k2" in total_transforms else 0.0,
        k3=total_transforms["k3"] if "k3" in total_transforms else 0.0,
        k4=total_transforms["k4"] if "k4" in total_transforms else 0.0,
        p1=total_transforms["p1"] if "p1" in total_transforms else 0.0,
        p2=total_transforms["p2"] if "p2" in total_transforms else 0.0,
    )
    distortion_array = np.array([distortion_params["k1"], distortion_params["k2"], distortion_params["k3"],
                                 distortion_params["k4"], distortion_params["p1"], distortion_params["p2"]])
    distortion = np.array(
        [distortion_array[0], distortion_array[1], distortion_array[4], distortion_array[5], distortion_array[2]])
    # K matrix
    K = np.zeros((3, 4))
    K[0, 0] = total_transforms["fl_x"]
    K[0, 2] = total_transforms["cx"]
    K[1, 1] = total_transforms["fl_y"]
    K[1, 2] = total_transforms["cy"]
    K[2, 2] = 1
    # Generate rgb paths
    fnames = []
    for frame in transforms["frames"]:
        filepath = frame["file_path"]
        fnames.append(os.path.join(dir, filepath))
    inds = np.argsort(fnames)
    frames = [transforms["frames"][ind] for ind in inds]
    frame_name_total = []
    posewithname = []
    for framee in frames:
        frame_pose = framee["transform_matrix"]
        frame_name = framee["file_path"]
        # frame_pose_total.append(frame_pose)
        frame_name_total.append(os.path.join(dir, frame_name))
        posewithname.append({"frame": frame_pose, "name": frame_name})
    # (W,H)
    W = total_transforms["w"]
    H = total_transforms["h"]
    wh = {"w":W, "h":H}
    return (frame_name_total, K, distortion, posewithname, wh)

def nerfacto2opencv(poses_dict: dict) -> dict:
    """ convert Nerfacto(OpenGL) poses(x,y,z) to the Opencv system(x,-y,-z) then change
    "c2w" into "w2c".
    :param poses_dict: poses in OpenGL system --> Dict
    :return: posewithname: world to camera in OpenCV system with image names
    """
    posewithname = []
    gather = []
    for subject in poses_dict: gather.append(subject["frame"])
    poses = np.stack(gather)
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, axis=0)
    translation = mean_origin
    # Align translation column with z axis
    up = np.mean(poses[:, :3, 1], axis=0)
    up = up / np.linalg.norm(up)
    rotation = rotation_matrix(up, np.array([0, 0, 1]))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], axis=-1)
    oriented_poses = transform @ poses
    poses = oriented_poses
    scale_factor = 1.0
    poses[:, 0, 3] += 1.0  # we added +1 in the x axis of translation column
    scale_factor /= float(np.max(np.abs(poses[:, :3, 3])))
    poses[:, :3, 3] *= scale_factor
    # Convert from COLMAP's camera coordinate system to ours+
    poses = np.concatenate((poses, np.tile(np.array([[0, 0, 0, 1]]), (int(len(poses)), 1, 1))), 1)
    poses[:, 0:3, 1:3] *= -1
    w2c = np.linalg.inv(poses)  # Change c2w to w2c

    # Add image name into W2C
    for key, value in zip(w2c, poses_dict):
        posewithname.append({"frame": key, "name": value["name"]})
    return posewithname

if __name__ == '__main__':
    runtool()

def get_preop(dir):
    dir = dir / "gt_mesh/rigid/preop/higado_simple2.obj"
    verts_preop, faces_preop, properties_preop = load_obj(dir,load_textures=True, create_texture_atlas=True)
    return verts_preop, faces_preop


##############################  Resize images ###################################
def resize():
    import cv2
    import os

    input_folder = "/home/geintra/MIS/data/liver/liver_real/20180905/output_total/circle_mask"
    output_folder = "/home/geintra/MIS/data/liver/liver_real/20180905/output_total/circle_mask_8"
    new_size = (240, 135)

    os.makedirs(output_folder, exist_ok=True)

    # Get sorted list to preserve order
    filenames = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # Loop with counter
    for idx, filename in enumerate(filenames, start=1):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            resized_img = cv2.resize(
                img, new_size, interpolation=cv2.INTER_LANCZOS4
            )

            # Create new name: circle_001.png, circle_002.png, ...
            new_name = f"circle_{idx:03d}.png"

            # Save single channel if needed
            cv2.imwrite(
                os.path.join(output_folder, new_name),
                resized_img[:, :, 0:1]
            )

            print(f"Saved: {new_name}")

    print("Resizing completed!")




#
# import json
# import re
#
# input_path = "/home/geintra/MIS/data/knee/Dataset_unity/hermine/transforms.json"
#
# output_path = "//home/geintra/MIS/data/knee/Dataset_unity/hermine/transforms_frame.json"
#
# with open(input_path, "r") as f:
#
#     data = json.load(f)
#
#
#
# def transform(path):
#
#     if path is None:
#
#         return path
#
#     # color -> frame
#
#     path = path.replace("color", "frame")
#
#     # find index
#
#     m = re.search(r'_(\d+)(\.png)', path)
#
#     if not m:
#
#         return path
#
#     num = int(m.group(1)) + 1  # shift +1 so start becomes 00001
#
#     new = f"_{num:05d}{m.group(2)}"
#
#     path = re.sub(r'_(\d+)(\.png)', new, path)
#
#     return path
#
#
#
# for fr in data.get("frames", []):
#
#     if "file_path" in fr:
#
#         fr["file_path"] = transform(fr["file_path"])
#
#     if "mask_path" in fr:
#
#         fr["mask_path"] = transform(fr["mask_path"])
#
#
#
# with open(output_path, "w") as f:
#
#     json.dump(data, f, indent=2)
#
#
#
