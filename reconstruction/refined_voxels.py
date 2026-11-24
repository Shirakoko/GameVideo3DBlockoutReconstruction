"""Observed static surfaces -> bounded, flat-colored editable voxel shells.

No cube objects per pixel, no inferred solid interiors, no video projection.
Only external grid faces are exported, chunked by semantic category.
"""
from io import BytesIO
import hashlib

import numpy as np
import open3d as o3d
from PIL import Image

from reconstruction.coordinate_transform import transform_points
from reconstruction.refined_calibration import unproject_inferred, project
from reconstruction.refined_segmentation import CATEGORIES


class VoxelFusion:
    def __init__(self, k, size=.2, *, stride=3, max_cells=400000):
        if not np.isfinite(size) or not .05 <= size <= 1.:
            raise ValueError("Voxel block size must be in [0.05, 1.0] metres")
        self.k, self.size, self.stride, self.max_cells = k, size, stride, max_cells
        self.frames, self.records = [], []
        self.seen, self.duplicates = {}, {}
        self.report = {}

    def integrate(self, rgb, depth, valid, labels, pose, frame_index):
        signature = hashlib.sha256()
        for data in (rgb, depth, valid, labels, pose):
            signature.update(np.ascontiguousarray(data).tobytes())
        digest = signature.hexdigest()
        if digest in self.seen:
            self.duplicates[int(frame_index)] = self.seen[digest]
            return
        self.seen[digest] = int(frame_index)
        local, uv = unproject_inferred(depth, self.k, valid, stride=self.stride)
        if not len(local):
            return
        world = transform_points(local, pose)
        category = labels[uv[:, 1], uv[:, 0]]
        keep = category != CATEGORIES.index("person")
        keys = np.column_stack((np.floor(world[keep]/self.size).astype(np.int64), category[keep]))
        colors = rgb[uv[keep, 1], uv[keep, 0]].astype(np.float64)/255
        unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), 3))
        np.add.at(sums, inverse, colors)
        # Each independent frame contributes one vote, not thousands of pixels.
        self.records.append((unique, sums/counts[:, None]))
        self.frames.append(int(frame_index))

    def extract(self, *, minimum_views=2):
        if not self.records:
            raise ValueError("No observed static voxel samples")
        keys = np.concatenate([r[0] for r in self.records])
        colors = np.concatenate([r[1] for r in self.records])
        unique, inverse, views = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), 3))
        np.add.at(sums, inverse, colors)
        keep = views >= minimum_views
        unique, colors, views = unique[keep], sums[keep]/views[keep, None], views[keep]
        if not len(unique):
            raise ValueError("No voxels supported by multiple training views")
        # One semantic owner per occupied cell, greatest independent evidence.
        order = np.lexsort((-views, unique[:, 2], unique[:, 1], unique[:, 0]))
        ordered = unique[order]
        first = np.r_[True, np.any(np.diff(ordered[:, :3], axis=0), axis=1)]
        chosen = order[first]
        unique, colors = unique[chosen], colors[chosen]
        if not len(unique):
            raise ValueError("No voxels supported by multiple training views")
        if len(unique) > self.max_cells:
            raise ValueError(f"Voxel cell budget exceeded ({len(unique)} > {self.max_cells}); increase --voxel-size")
        occupied = {tuple(key) for key in unique[:, :3]}
        meshes = {}
        for category in np.unique(unique[:, 3]):
            selected = unique[:, 3] == category
            mesh = voxel_shell(unique[selected, :3], colors[selected], self.size, occupied)
            if len(mesh.triangles):
                meshes[CATEGORIES[int(category)]] = mesh
        self.report = dict(style="voxel", block_size_metres=self.size, occupied_cells=len(unique),
            minimum_independent_views=minimum_views, geometry_training_frames=sorted(self.frames),
            duplicate_frame_aliases=self.duplicates,
            surface_only=True, internal_faces_removed=True, hidden_surface_completion=False,
            palette="4096 quantized observed RGB colors; flat per face", provenance="inferred")
        return meshes


def voxel_shell(cells, colors, size, occupied=None):
    cells = np.asarray(cells, np.int64)
    colors = np.asarray(colors, float)
    occupied = occupied if occupied is not None else {tuple(c) for c in cells}
    # Outward winding; duplicated corners preserve sharp edges and face colors.
    sides = [((-1, 0, 0), [(0,0,0),(0,0,1),(0,1,1),(0,1,0)], .80),
             ((1, 0, 0), [(1,0,0),(1,1,0),(1,1,1),(1,0,1)], .92),
             ((0, -1, 0), [(0,0,0),(1,0,0),(1,0,1),(0,0,1)], .86),
             ((0, 1, 0), [(0,1,0),(0,1,1),(1,1,1),(1,1,0)], .95),
             ((0, 0, -1), [(0,0,0),(0,1,0),(1,1,0),(1,0,0)], .72),
             ((0, 0, 1), [(0,0,1),(1,0,1),(1,1,1),(0,1,1)], 1.)]
    vertices, triangles, face_colors = [], [], []
    for cell, color in zip(cells, colors):
        for offset, corners, shade in sides:
            if tuple(cell+offset) in occupied:
                continue
            start = len(vertices)
            vertices.extend((cell+np.asarray(corners))*size)
            triangles.extend(((start, start+1, start+2), (start, start+2, start+3)))
            face_colors.extend([np.clip(color*shade, 0, 1)]*4)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices).reshape(-1, 3)),
        o3d.utility.Vector3iVector(np.asarray(triangles, np.int32).reshape(-1, 3)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(face_colors).reshape(-1, 3))
    mesh.compute_vertex_normals()
    return mesh


def palette_mesh(mesh):
    triangles = np.asarray(mesh.triangles)
    points = np.asarray(mesh.vertices)[triangles].reshape(-1, 3)
    colors = np.asarray(mesh.vertex_colors)[triangles].mean(1)
    quantized = np.rint(np.clip(colors, 0, 1)*15).astype(int)
    slots = quantized[:, 0]*256+quantized[:, 1]*16+quantized[:, 2]
    uv = np.repeat(np.column_stack(((slots % 64+.5)/64, (slots//64+.5)/64)), 3, axis=0)
    ids = np.arange(4096)
    palette = np.column_stack((ids//256, ids//16 % 16, ids % 16)).reshape(64, 64, 3)*17
    encoded = BytesIO()
    Image.fromarray(palette.astype(np.uint8)).save(encoded, format="PNG")
    return points, np.arange(len(points), dtype=np.uint32).reshape(-1, 3), uv, encoded.getvalue()


def observation_views(points, category, frames, frame_loader, poses, k):
    """Find one genuinely observed representative view of this world chunk."""
    center = np.mean(points, axis=0)
    sample = points[::max(1, len(points)//5000)]
    for index in sorted(frames, key=lambda i: np.linalg.norm(poses[i][:3,3]-center)):
        rgb, depth, valid, labels = frame_loader(index)
        uv, z = project(sample, poses[index], k)
        h,w = depth.shape
        finite = np.isfinite(uv).all(1) & (z > .1)
        xy = np.rint(np.where(finite[:,None], uv, -1)).astype(int)
        inside = finite & (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        x,y = np.clip(xy[:,0],0,w-1),np.clip(xy[:,1],0,h-1)
        visible = inside & valid[y,x] & (labels[y,x] == CATEGORIES.index(category))
        visible &= np.abs(depth[y,x]-z) <= .4+.025*z
        if visible.sum() < 4:
            continue
        x,y = x[visible],y[visible]
        mask = np.zeros((h,w),bool)
        rectangle = np.s_[y.min():y.max()+1,x.min():x.max()+1]
        mask[rectangle] = valid[rectangle] & (labels[rectangle] == CATEGORIES.index(category))
        if mask.sum() >= 4:
            return [index], {index:mask}
    raise ValueError("Voxel chunk has no visible representative source observation")
