"""Simple deterministic unit primitives, using data API (no UI operators)."""
import math

import bpy
from mathutils import Quaternion


COLORS = {"ground": (0.42, 0.45, 0.43, 1), "person": (0.72, 0.69, 0.64, 1),
          "tree": (0.42, 0.52, 0.44, 1), "rock": (0.55, 0.56, 0.59, 1), "barrier": (0.65, 0.59, 0.49, 1),
          "road": (0.30, 0.32, 0.34, 1), "wall": (0.61, 0.61, 0.59, 1), "building": (0.49, 0.51, 0.54, 1)}
COLORS.update(grass=(.36, .44, .32, 1), shrub=(.38, .43, .33, 1), environment=(.5, .5, .5, 1))


def material(category):
    mat = bpy.data.materials.new("Blockout_"+category)
    mat.diffuse_color = COLORS[category]
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = COLORS[category]
    shader.inputs["Roughness"].default_value = 0.85
    shader.inputs["Metallic"].default_value = 0
    return mat


def primitive_mesh(kind, name):
    if kind in {"box", "plane"}:
        # A measured plane has positive observed Z extent: use a thin slab,
        # not a zero-thickness mesh that silently loses the declared bounds.
        vertices = [(x*.5, y*.5, z*.5) for x, y, z in
                    [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                     (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    else:
        segments = 16
        if kind == "cylinder":
            rings = [(-.5, .5), (.5, .5)]
        elif kind == "capsule":
            # Normalized capsule with ellipsoidal endcaps after XYZ scaling.
            rings = [(-.25+.25*math.sin(-math.pi/2+i*math.pi/12), .5*math.cos(-math.pi/2+i*math.pi/12)) for i in range(7)]
            rings += [(.25+.25*math.sin(i*math.pi/12), .5*math.cos(i*math.pi/12)) for i in range(7)]
        else:
            raise ValueError("Unsupported primitive: "+kind)
        vertices = [(radius*math.cos(2*math.pi*j/segments), radius*math.sin(2*math.pi*j/segments), z)
                    for z, radius in rings for j in range(segments)]
        faces = [tuple(reversed(range(segments))), tuple((len(rings)-1)*segments+j for j in range(segments))]
        for i in range(len(rings)-1):
            for j in range(segments):
                n = (j+1) % segments
                faces.append((i*segments+j, i*segments+n, (i+1)*segments+n, (i+1)*segments+j))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def apply_transform(obj, transform, dimensions, *, mesh=False):
    obj.location = transform["position"]
    x, y, z, w = transform["rotation"]
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Quaternion((w, x, y, z))
    obj.scale = transform["scale"] if mesh else [d*s for d, s in zip(dimensions, transform["scale"])]


def import_mesh(item, collection):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=item["mesh_asset"]["path"])
    created = set(bpy.data.objects)-before
    meshes = [o for o in created if o.type == "MESH"]
    if len(meshes) != 1:
        raise ValueError("Canonical refined GLB must contain exactly one mesh object")
    obj = meshes[0]
    # Apply importer axis/root transforms to vertices once. JSON transform is
    # the independent local-metre to world placement, not glTF's hierarchy.
    obj.data.transform(obj.matrix_world.copy())
    obj.parent = None
    from mathutils import Matrix
    obj.matrix_world = Matrix.Identity(4)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)
    for extra in created-{obj}:
        bpy.data.objects.remove(extra, do_unlink=True)
    for image in bpy.data.images:
        if image.source == "FILE" and image.has_data and not image.packed_file:
            image.pack()
    return obj


def create_object(item, collection, mat):
    name = item["object_id"]
    if item["primitive_type"] == "mesh":
        obj = import_mesh(item, collection)
        obj.name = name
    else:
        obj = bpy.data.objects.new(name, primitive_mesh(item["primitive_type"], name+"_mesh"))
        collection.objects.link(obj)
        obj.data.materials.append(mat)
    obj["source_object_id"], obj["category"], obj["primitive_type"] = name, item["category"], item["primitive_type"]
    obj["is_static"] = "unknown" if item["is_static"] is None else str(item["is_static"])
    obj["dimensions_unscaled_m"] = item["dimensions"]
    obj["parent_id"] = item["parent_id"] or ""
    apply_transform(obj, item["transform"], item["dimensions"], mesh=item["primitive_type"] == "mesh")
    return obj
