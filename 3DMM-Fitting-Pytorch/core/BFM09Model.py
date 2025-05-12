import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from core.BaseModel import BaseReconModel
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)
import matplotlib.pyplot as plt
import cv2


class BFM09ReconModel(BaseReconModel):
    def __init__(self, model_dict, **kargs):
        super(BFM09ReconModel, self).__init__(**kargs)

        self.skinmask = torch.tensor(
            model_dict['skinmask'], requires_grad=False, device=self.device)

        self.kp_inds = torch.tensor(
            model_dict['keypoints']-1).squeeze().long().to(self.device)

        self.meanshape = torch.tensor(model_dict['meanshape'],
                                      dtype=torch.float32, requires_grad=False,
                                      device=self.device)

        self.idBase = torch.tensor(model_dict['idBase'],
                                   dtype=torch.float32, requires_grad=False,
                                   device=self.device)

        self.expBase = torch.tensor(model_dict['exBase'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        self.meantex = torch.tensor(model_dict['meantex'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        self.texBase = torch.tensor(model_dict['texBase'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        self.tri = torch.tensor(model_dict['tri']-1,
                                dtype=torch.int64, requires_grad=False,
                                device=self.device)

        self.point_buf = torch.tensor(model_dict['point_buf']-1,
                                      dtype=torch.int64, requires_grad=False,
                                      device=self.device)

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
        exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
        # ruler angles(x,y,z) for rotation of dim 3
        angles = coeffs[:, 224:227]
        # lighting coeff for 3 channel SH function of dim 27
        gamma = coeffs[:, 227:254]
        translation = coeffs[:, 254:]  # translation coeff of dim 3

        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation):
        coeffs = torch.cat([id_coeff, exp_coeff, tex_coeff,
                            angles, gamma, translation], dim=1)
        return coeffs

    def forward(self, coeffs, render=True, finalSet=False):
        batch_num = coeffs.shape[0]

        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation = self.split_coeffs(
            coeffs)

        vs = self.get_vs(id_coeff, exp_coeff)

        if finalSet:
            R, T = look_at_view_transform(10, 0, 0)  # camera's position
            cameras = FoVPerspectiveCameras(device=torch.device('cpu'), R=R, T=T, znear=0.01,
                                            zfar=50,
                                            fov=2*np.arctan(self.img_size//2/1015)*180./np.pi)

            lights = PointLights(device=torch.device('cpu'), location=[[0.0, 0.0, 1e5]],
                                ambient_color=[[1, 1, 1]],
                                specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

            raster_settings = RasterizationSettings(
                image_size=self.img_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            blend_params = blending.BlendParams(background_color=[1, 1, 1]) #NOTE: this is where you can change background.

            my_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=torch.device('cpu'),
                    cameras=cameras,
                    lights=lights,
                    blend_params=blend_params
                )
            )
            ROTATION_ANGLES = [0.8, -0.85, 0, 0.44, -0.44] #Rotation angles, in radians.
            rotation_tensor_list = [torch.tensor([[0,z_angle,0]], dtype=torch.float32, requires_grad=True, device='cpu') for z_angle in ROTATION_ANGLES]
            rotation_list = [self.compute_rotation_matrix(angles) for angles in rotation_tensor_list] #NOTE: diff data type than in the else clause.
            vs_t_list = [self.rigid_transform(vs, rotation, translation) for rotation in rotation_list]
            lms_t_list = [self.get_lms(vs_t) for vs_t in vs_t_list]
            lms_proj_list = [self.project_vs(lms_t) for lms_t in lms_t_list]
            lms_proj_list_2 = [torch.stack(
                [lms_proj[:, :, 0], self.img_size-lms_proj[:, :, 1]], dim=2) for lms_proj in lms_proj_list]

            if render:
                face_texture = self.get_color(tex_coeff)
                face_norm_l = self.compute_norm(vs, self.tri, self.point_buf)
                face_norm_r_list = [face_norm_l.bmm(rotation_list[i]) for i in range(len(ROTATION_ANGLES))]
                
                face_color_list = [self.add_illumination(
                    face_texture, face_norm_r, gamma) for face_norm_r in face_norm_r_list]

                face_color_tv_list = [TexturesVertex(face_color) for face_color in face_color_list]
                mesh_list = [Meshes(vs_t_list[i], self.tri.repeat(batch_num, 1, 1), face_color_tv_list[i]) for i in range(len(ROTATION_ANGLES))]
                for i in range(len(mesh_list)):
                    rendered_angled_img = my_renderer(mesh_list[i])
                    rendered_angled_img = torch.clamp(rendered_angled_img, 0, 255)
                    cv2.imwrite(f"angled_img_{i}.jpg", rendered_angled_img[0, ..., :3].cpu().numpy())

        rotation = self.compute_rotation_matrix(angles)
        vs_t = self.rigid_transform(
            vs, rotation, translation)

        lms_t = self.get_lms(vs_t)
        lms_proj = self.project_vs(lms_t)
        lms_proj = torch.stack(
            [lms_proj[:, :, 0], self.img_size-lms_proj[:, :, 1]], dim=2)
        if render:
            face_texture = self.get_color(tex_coeff)
            face_norm = self.compute_norm(vs, self.tri, self.point_buf)
            face_norm_r = face_norm.bmm(rotation)
            face_color = self.add_illumination(
                face_texture, face_norm_r, gamma)
            face_color_tv = TexturesVertex(face_color)

            mesh = Meshes(vs_t, self.tri.repeat(
                batch_num, 1, 1), face_color_tv)
            rendered_img = self.renderer(mesh)
            rendered_img = torch.clamp(rendered_img, 0, 255)

            return {'rendered_img': rendered_img,
                    'lms_proj': lms_proj,
                    'face_texture': face_texture,
                    'vs': vs_t,
                    'tri': self.tri,
                    'color': face_color}
        else:
            return {'lms_proj': lms_proj}

    def get_vs(self, id_coeff, exp_coeff):
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape

        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - \
            self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def get_color(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum(
            'ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture

    def get_skinmask(self):
        return self.skinmask

    def init_coeff_dims(self):
        self.id_dims = 80
        self.tex_dims = 80
        self.exp_dims = 64
