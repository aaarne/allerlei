import numpy as np
from functools import reduce
from .tools import Timer, TUMColors, TUMColor
import pymesh


def join_meshes(meshes):
    meshes = [*filter(lambda m: m is not None, meshes)]
    if len(meshes) < 1:
        raise ValueError("No meshes to join.")
    assert np.all([mesh.dim == meshes[0].dim for mesh in meshes])
    vertex_indices = np.cumsum([0, *[mesh.num_vertices for mesh in meshes]])
    face_indices = np.cumsum([0, *[mesh.num_faces for mesh in meshes]])
    vertices = np.empty((vertex_indices[-1], meshes[0].dim))
    faces = np.empty((face_indices[-1], meshes[0].nodes_per_element))
    for mesh, vi, vj, fi, fj in zip(
            meshes,
            vertex_indices,
            vertex_indices[1::],
            face_indices,
            face_indices[1::]):
        vertices[vi:vj, :] = mesh.vertices
        faces[fi:fj, :] = mesh.faces + vi

    big_mesh = pymesh.form_mesh(vertices, faces)

    for prop in reduce(lambda s1, s2: s1 & s2, map(set, [mesh.get_attribute_names() for mesh in meshes])):
        big_mesh.add_attribute(prop)
        big_mesh.set_attribute(prop, np.concatenate([mesh.get_attribute(prop) for mesh in meshes]))

    return big_mesh


def plot_mesh(mesh=None, verts=None, faces=None, ax=None, facecolor=(1, 0, 0, 0.2), edgecolor=(0, 0, 0, 0),
              optimize_scaling=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if verts is None:
        if mesh is None:
            raise ValueError
        else:
            verts = mesh.vertices
            faces = mesh.faces
    if ax is None:
        f = plt.figure()
        a = f.add_subplot(111, projection='3d')
    else:
        a = ax
    pltmesh = Poly3DCollection(verts[faces])
    pltmesh.set_edgecolor(edgecolor)
    pltmesh.set_facecolor(facecolor)
    a.add_collection(pltmesh)
    for i, f in zip(range(3), [a.set_xlabel, a.set_ylabel, a.set_zlabel]):
        f(f"$q_{i+1}$")

    if optimize_scaling:
        a.set_aspect("auto")
        for i, func in enumerate([a.set_xlim, a.set_ylim, a.set_zlim]):
            func(np.min(verts[:, i]), np.max(verts[:, i]))
    if ax is None:
        plt.show()


def smooth_mesh(mesh, n_runs=1):
    if n_runs == 0:
        return mesh

    mesh.enable_connectivity()
    n = mesh.vertices.shape[0]
    pos = mesh.vertices.copy()
    final = np.empty((n, 3))

    with Timer("Smoothing Mesh"):
        for _ in range(n_runs):
            for i in range(n):
                final[i] = np.mean([pos[j] for j in mesh.get_vertex_adjacent_vertices(i)], axis=0)
            pos = final.copy()

    new_mesh = pymesh.form_mesh(final, mesh.faces)
    for prop in mesh.get_attribute_names():
        new_mesh.add_attribute(prop)
        new_mesh.set_attribute(prop, mesh.get_attribute(prop))

    return new_mesh


def enhance_mesh(m, rel_target_edge_length=0.33):
    import pymesh
    with Timer("Enhancing mesh:"):
        with Timer("Removing duplicated vertices"):
            m, _ = pymesh.remove_duplicated_vertices(m)
        with Timer("Removing duplicated faces"):
            m, _ = pymesh.remove_duplicated_faces(m)
        with Timer("Removing degenerated triangles"):
            m, _ = pymesh.remove_degenerated_triangles(m)
        with Timer("Removing obtuse triangles"):
            m, _ = pymesh.remove_obtuse_triangles(m)
        with Timer("Collapsing short edges"):
            m, _ = pymesh.collapse_short_edges(m, rel_threshold=rel_target_edge_length)
    return m


def colorize_mesh(mesh, color=None, r=None, b=None, g=None):
    mesh.add_attribute('color')
    c = np.empty_like(mesh.vertices)
    if color:
        if type(color) is TUMColor:
            c[:, :] = TUMColors.hextofloats(color)
        else:
            c[:, :] = color
    elif r and b and g:
        c[:, :] = (r, b, g)
    else:
        raise ValueError("Illegal Color specification")
    mesh.set_attribute('color', c)


