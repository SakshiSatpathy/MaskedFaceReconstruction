import os
import torch
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# add path for demo utils functions 
import sys

"""
This file served as the original approach taken to load textures from .obj output of 3DMM.
Originally, I modified the 3DMM-Fitting-Pytorch code to output a textured image. These attempts would fail due to incorrectly named variables.
Additionally, Pytorch3D requires .mtl files to load the textured mesh, which I was unable to find outputted in the 3DMM-Fitting-Pytorch.
This code was made to load the outputted mesh through Open3D, and load the textured image on top of it, and then convert it to Pytorch3D.
This conversion would also lead to several errors. In the end, I finally found the correctly named rendered image from 3DMM-Fitting-Pytorch, and outputted it directly, effectively deprecating this file.
"""



device = torch.device("cpu")
obj_filename = "3DMM-Fitting-Pytorch/results/mask_attempt_8_mesh.obj"
texture_img_filename = "3DMM-Fitting-Pytorch/results/mask_attempt_8_texture_img.jpeg"

"""
O3D Approach
"""
#Load the mesh from a file
mesh = o3d.io.read_triangle_mesh(obj_filename)
#Check if the mesh has vertex normals and compute them if missing
if not mesh.has_vertex_normals():
  mesh.compute_vertex_normals()
#Display the mesh before applying any texture
#o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)


#Load the texture image
texture_image = o3d.io.read_image(texture_img_filename)

#Assign the texture image to the mesh
mesh.textures = [texture_image]

#Display the textured mesh
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
###NOTE: the o3d drawing of geometries is the same as when the texture isn't added.

#Save the textured mesh to an output file (ex: .obj format)
#o3d.io.write_triangle_mesh("output_mesh_with_texture.obj", mesh)
#Warning: Write OBJ cannot include triangle normals



##############
#### Converting Open3D Mesh into Pytorch Mesh
##############
vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
faces = torch.tensor(np.array(mesh.triangles), dtype=torch.int64)

print(f"Shape of vertices: {vertices.shape}")
print(f"Shape of faces: {faces.shape}")

##NEEDS textures for rendering -- this would fail, as mentioned above, due to incorrect outputted images from the 3DMM-Fitting-Pytorch
if mesh.has_textures():
    image_o3d = cv2.imread(texture_img_filename)
    #### PROBLEM: image has one channel.
    texture_np = np.asarray(image_o3d)

    print(f"The sum of all elements in the textured image is: {texture_np.sum()}") # currently nonempty.
    print(f"The dimension of the texture array is: {texture_np.shape}") # This outputs 256, 256, 3

    texture_tensor = torch.tensor(texture_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
else:
    texture_tensor = None

if texture_tensor is not None:
    textures = TexturesUV(verts_uvs=[torch.zeros_like(vertices)[:, :2]], faces_uvs=[faces], maps=texture_tensor)
    mesh_pytorch = Meshes(verts=[vertices], faces=[faces], textures=textures)
else:
    mesh_pytorch = Meshes(verts=[vertices], faces=[faces])

#verts, faces, aux = load_obj(obj_filename)
#meshes = Meshes(verts=[verts], faces=[faces.verts_idx])

R, T = look_at_view_transform(2.7, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)
meshes.textures = torch.ones_like(verts)[None]
images = renderer(mesh_pytorch) 
#Attribute Error here: 'open3d.cpu.pybind.geometry.TriangleMesh' object has no attribute 'verts_padded'
#It seems that the root of the problem is that I am passing an Open3D Mesh into a Pytorch function; I need a Pytorch Mesh 
alpha = images[..., 3]
mask = (alpha > 0).cpu().numpy()[0]

if not cv2.imwrite("binary_mask.png", mask):
    print("ERROR: Unable to output Binary Mask image")
else:
    cv2.imwrite("binary_mask.png", mask)


plt.figure(figsize=(10, 10))

#plt.savefig("rendered_mesh2.png")
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.close()
