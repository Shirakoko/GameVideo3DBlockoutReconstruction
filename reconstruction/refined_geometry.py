"""Streaming TSDF/thin surfaces and canonical, textured local-metre GLBs."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import struct

import numpy as np
import open3d as o3d
from PIL import Image

from reconstruction.coordinate_transform import inverse_rigid, transform_points
from reconstruction.refined_calibration import project
from reconstruction.refined_segmentation import CATEGORIES

THIN = {"tree", "barrier", "grass", "shrub"}


def depth_surface(depth, valid, k, pose, *, stride=4, relative_jump=.025, absolute_jump=.12):
    """Never connect across invalid masks, foreground edges or depth gaps."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[v, u]
    mask = valid[v, u] & np.isfinite(z) & (z > 0)
    ids = np.arange(z.size).reshape(z.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    triangles = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3), np.stack((b, c, d), -1).reshape(-1, 3)))
    zz = z.ravel()[triangles]
    keep = mask.ravel()[triangles].all(1) & ((zz.max(1)-zz.min(1)) <= absolute_jump+relative_jump*zz.min(1))
    triangles = triangles[keep]
    if not len(triangles):
        return o3d.geometry.TriangleMesh()
    used, inverse = np.unique(triangles, return_inverse=True)
    uv = np.column_stack((u.ravel()[used], v.ravel()[used]))
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(k).T
    points = transform_points(rays*z.ravel()[used, None], pose)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(inverse.reshape(-1, 3)))
    mesh.compute_vertex_normals()
    return mesh


def deduplicate_faces(mesh, voxel=.04):
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(triangles):
        return mesh
    center = vertices[triangles].mean(1)
    normal = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]], vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    keys = np.column_stack((np.floor(center/voxel).astype(np.int64), np.rint(normal*4).astype(np.int64)))
    _, selected = np.unique(keys, axis=0, return_index=True)
    mesh.triangles = o3d.utility.Vector3iVector(triangles[np.sort(selected)])
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


class SurfaceFusion:
    def __init__(self, k, voxel=.04, truncation=.12):
        self.k = np.asarray(k)
        self.voxel = voxel
        self.volumes, self.thin = {}, {}
        self.truncation = truncation

    def integrate(self, rgb, depth, valid, labels, pose):
        for category in CATEGORIES[:-1]:
            category_mask = valid & (labels == CATEGORIES.index(category))
            if category_mask.sum() < 48:
                continue
            if category in THIN:
                mesh = depth_surface(depth, category_mask, self.k, pose)
                if len(mesh.triangles):
                    self.thin.setdefault(category, o3d.geometry.TriangleMesh())
                    self.thin[category] += mesh
            else:
                if category not in self.volumes:
                    self.volumes[category] = o3d.pipelines.integration.ScalableTSDFVolume(
                        voxel_length=self.voxel, sdf_trunc=self.truncation,
                        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
                volume = self.volumes[category]
                # Nearest sample avoids inventing depth values at silhouettes.
                small_depth = np.where(category_mask, depth, 0)[::4, ::4].astype(np.float32)
                small_rgb = rgb[::4, ::4].copy()
                hh, ww = small_depth.shape
                kk = self.k.copy()
                kk[:2] /= 4
                intrinsic = o3d.camera.PinholeCameraIntrinsic(ww, hh, kk[0, 0], kk[1, 1], kk[0, 2], kk[1, 2])
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(small_rgb),
                    o3d.geometry.Image(small_depth), depth_scale=1., depth_trunc=80., convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic, inverse_rigid(pose))

    def extract(self, max_faces=600000):
        meshes = {category: volume.extract_triangle_mesh() for category, volume in self.volumes.items()}
        meshes.update({category: deduplicate_faces(mesh, self.voxel) for category, mesh in self.thin.items()})
        for category, mesh in meshes.items():
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            if len(mesh.triangles) > max_faces:
                mesh = mesh.simplify_quadric_decimation(max_faces)
                meshes[category] = mesh
            mesh.compute_vertex_normals()
        return {category: mesh for category, mesh in meshes.items() if len(mesh.triangles)}


def split_chunks(mesh, size=8., max_faces=50000, *, preserve_voxels=False, voxel_size=.2):
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    centers = vertices[triangles].mean(1)
    if preserve_voxels:
        normals = np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]], vertices[triangles[:,2]]-vertices[triangles[:,0]])
        normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        # Move the face center into its owning cube, then chunk by that cube's
        # center. Boundary faces must not become zero-thickness orphan chunks.
        centers = (np.floor((centers-normals*voxel_size*.25)/voxel_size)+.5)*voxel_size
    keys = np.floor(centers/size).astype(int)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    for i, key in enumerate(unique):
        selected = triangles[inverse == i]
        if len(selected) < 4 and not preserve_voxels:
            continue
        used, remap = np.unique(selected, return_inverse=True)
        part = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices[used]),
            o3d.utility.Vector3iVector(remap.reshape(-1, 3)))
        if mesh.has_vertex_colors():
            part.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used])
        if len(part.triangles) > max_faces and not preserve_voxels:
            part = part.simplify_quadric_decimation(max_faces)
        part.compute_vertex_normals()
        yield tuple(map(int, key)), part


def bake_texture(mesh, frames, frame_loader, poses, k, *, max_views=4):
    """Select jointly visible source views per face, rejecting dynamic masks.

    No holdout view is supplied by the caller. Texture and geometry remain
    static across output frames; the current original frame is never overlaid.
    """
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    centers = vertices[triangles].mean(1)
    normal = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]], vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    scores, views = [], []
    for index in frames:
        _, depth, valid, labels = frame_loader(index)
        uv, z = project(centers, poses[index], k)
        h, w = depth.shape
        finite = np.isfinite(uv).all(1) & (z > 0)
        safe_uv = np.where(finite[:, None], uv, -1)
        x, y = np.rint(safe_uv[:, 0]).astype(int), np.rint(safe_uv[:, 1]).astype(int)
        inside = finite & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        xx, yy = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
        visible = inside & valid[yy, xx] & (np.abs(depth[yy, xx]-z) <= .15+.025*z)
        visible &= labels[yy, xx] != CATEGORIES.index("person")
        ray = poses[index][:3, 3]-centers
        distance = np.maximum(np.linalg.norm(ray, axis=1), 1e-6)
        # Thin double-sided surfaces may have either normal orientation.
        angle = np.abs(np.sum(normal*ray/distance[:, None], axis=1))
        score = np.where(visible, (.1+angle)/distance, 0)
        if (score > 0).any():
            scores.append(score)
            views.append(index)
    if not scores:
        raise ValueError("No visible training image can texture this mesh chunk")
    scores = np.asarray(scores)
    # Greedy coverage picks a bounded atlas, not sixty full-frame textures.
    selected, best = [], np.zeros(len(triangles))
    for _ in range(min(max_views, len(views))):
        gain = np.maximum(scores-best, 0).sum(1)
        choice = int(np.argmax(gain))
        if gain[choice] <= 0:
            break
        selected.append(choice)
        best = np.maximum(best, scores[choice])
    if not selected:
        raise ValueError("Texture coverage unavailable")
    # Recompute selected scores because greedy best is only the coverage state.
    selected_scores = scores[selected]
    owner = np.argmax(selected_scores, axis=0)
    keep = selected_scores.max(0) > 0
    triangles, owner = triangles[keep], owner[keep]
    w, h = frame_loader(views[selected[0]])[0].shape[1], frame_loader(views[selected[0]])[0].shape[0]
    atlas = Image.new("RGB", (w*2, h*2), (90, 90, 90))
    expanded_vertices, expanded_uv = [], []
    selected_frames = [views[s] for s in selected]
    for slot, index in enumerate(selected_frames):
        atlas.paste(Image.fromarray(frame_loader(index)[0]), ((slot % 2)*w, (slot//2)*h))
    for slot, index in enumerate(selected_frames):
        face_ids = np.flatnonzero(owner == slot)
        points = vertices[triangles[face_ids]].reshape(-1, 3)
        uv, z = project(points, poses[index], k)
        good = (z.reshape(-1, 3) > 0).all(1) & np.isfinite(uv.reshape(-1, 3, 2)).all((1, 2))
        points, uv = points.reshape(-1, 3, 3)[good].reshape(-1, 3), uv.reshape(-1, 3, 2)[good].reshape(-1, 2)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)).reshape(-1, 3).all(1)
        points, uv = points.reshape(-1, 3, 3)[inside].reshape(-1, 3), uv.reshape(-1, 3, 2)[inside].reshape(-1, 2)
        uv[:, 0] = (uv[:, 0]+(slot % 2)*w)/(2*w)
        uv[:, 1] = (uv[:, 1]+(slot//2)*h)/(2*h)
        expanded_vertices.append(points)
        expanded_uv.append(uv)
    points = np.concatenate(expanded_vertices)
    uv = np.concatenate(expanded_uv)
    if len(points) < 3:
        raise ValueError("No complete visible triangles to bake")
    faces = np.arange(len(points), dtype=np.uint32).reshape(-1, 3)
    encoded = BytesIO()
    atlas.save(encoded, format="PNG")
    return points, faces, uv, encoded.getvalue(), dict(source_frames=selected_frames,
        textured_faces=len(faces), rejected_faces=int(len(np.asarray(mesh.triangles))-len(faces)),
        texture_mode="observed_baked_rgb", relighting=False, hidden_surface_completion=False)


def write_glb(path, vertices, triangles, uv, png):
    """One mesh, metres, glTF Y-up; one embedded unlit RGB texture."""
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.uint32)
    uv = np.asarray(uv, dtype=np.float32)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or triangles.max() >= len(vertices) or uv.shape != (len(vertices), 2) or not np.isfinite(uv).all()):
        raise ValueError("Invalid canonical mesh arrays")
    # Blender Z-up -> glTF Y-up. Blender's importer performs the inverse.
    gltf_vertices = vertices[:, [0, 2, 1]].copy()
    gltf_vertices[:, 2] *= -1
    chunks = [gltf_vertices.astype("<f4").tobytes(), uv.astype("<f4").tobytes(),
              triangles.astype("<u4").tobytes(), png]
    binary, views = bytearray(), []
    for index, chunk in enumerate(chunks):
        views.append(dict(buffer=0, byteOffset=len(binary), byteLength=len(chunk),
                          **({"target": 34962} if index < 2 else {"target": 34963} if index == 2 else {})))
        binary.extend(chunk)
        binary.extend(b"\0"*((-len(binary)) % 4))
    model = dict(asset=dict(version="2.0", generator="local inferred refined reconstruction"),
        extensionsUsed=["KHR_materials_unlit"], buffers=[dict(byteLength=len(binary))], bufferViews=views,
        accessors=[dict(bufferView=0, componentType=5126, count=len(vertices), type="VEC3", min=gltf_vertices.min(0).tolist(), max=gltf_vertices.max(0).tolist()),
                   dict(bufferView=1, componentType=5126, count=len(vertices), type="VEC2"),
                   dict(bufferView=2, componentType=5125, count=triangles.size, type="SCALAR")],
        images=[dict(bufferView=3, mimeType="image/png")], samplers=[dict(magFilter=9729, minFilter=9729, wrapS=33071, wrapT=33071)],
        textures=[dict(source=0, sampler=0)], materials=[dict(name="ObservedRGB", doubleSided=True,
            pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0, roughnessFactor=1), extensions={"KHR_materials_unlit": {}})],
        meshes=[dict(primitives=[dict(attributes=dict(POSITION=0, TEXCOORD_0=1), indices=2, material=0)])],
        nodes=[dict(mesh=0)], scenes=[dict(nodes=[0])], scene=0)
    document = json.dumps(model, separators=(",", ":")).encode()
    document += b" "*((-len(document)) % 4)
    data = struct.pack("<4sII", b"glTF", 2, 12+8+len(document)+8+len(binary))
    data += struct.pack("<I4s", len(document), b"JSON")+document
    data += struct.pack("<I4s", len(binary), b"BIN\0")+binary
    Path(path).write_bytes(data)
    inspect_glb(path)


def inspect_glb(path):
    data = Path(path).read_bytes()
    if len(data) < 28:
        raise ValueError("Truncated GLB")
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError("Invalid GLB header/length")
    size, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON" or size+28 > len(data):
        raise ValueError("Invalid GLB JSON chunk")
    model = json.loads(data[20:20+size])
    if len(model["meshes"]) != 1 or model["images"][0].get("mimeType") != "image/png":
        raise ValueError("Canonical GLB requires one mesh and an embedded PNG")
    bin_size, bin_kind = struct.unpack_from("<I4s", data, 20+size)
    if bin_kind != b"BIN\0" or 28+size+bin_size != len(data):
        raise ValueError("Invalid GLB binary chunk")
    for view in model["bufferViews"]:
        if view.get("byteOffset", 0) < 0 or view["byteLength"] < 0 or view.get("byteOffset", 0)+view["byteLength"] > bin_size:
            raise ValueError("GLB buffer view outside binary payload")
    return dict(vertices=model["accessors"][0]["count"], faces=model["accessors"][2]["count"]//3,
                embedded_texture=True, units="m", physical_geometry_verified=False)
