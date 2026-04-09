"""
Mesh quality evaluation via bidirectional Chamfer distance.

Compares one or more reconstructed mesh sets against a reference point cloud.
Uses a fixed per-object bounding box (union of all experiment meshes) so
recall numbers are comparable across experiments.

Usage:
    # compare multiple experiment output dirs against a reference point cloud
    python eval_mesh_quality.py \
        --ref input/gold_data_undistort/aggregated_pointcloud.ply \
        --meshes \
            original:input/gold_data_undistort/output/meshes \
            raw_depth:input/gold_data_undistort_rawdepth/output/meshes \
            depth1:input/gold_data_undistort_rawdepth_depth1/output/meshes

    # single experiment, useful for quick sanity check
    python eval_mesh_quality.py \
        --ref input/gold_data_undistort/aggregated_pointcloud.ply \
        --meshes myrun:input/myrun/output/meshes

Metrics:
    precision  mesh surface -> ref cloud  (accuracy where mesh exists, mm)
    recall     ref cloud -> mesh surface  (completeness, mm — lower is better)
    faces      number of mesh faces
"""

import argparse
import os
import numpy as np
import trimesh
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mesh quality against a reference point cloud.")
    parser.add_argument("--ref", required=True, help="Path to reference point cloud (.ply)")
    parser.add_argument("--meshes", nargs="+", required=True,
                        metavar="NAME:MESH_DIR",
                        help="One or more name:mesh_dir pairs (meshes named 1.obj, 2.obj, ...)")
    parser.add_argument("--objects", nargs="+", type=int, default=[1, 2, 3],
                        help="Object IDs to evaluate (default: 1 2 3)")
    parser.add_argument("--samples", type=int, default=20000,
                        help="Surface sample count per mesh (default: 20000)")
    parser.add_argument("--bbox-margin", type=float, default=0.01,
                        help="Margin added to bounding box when clipping ref cloud (default: 0.01m)")
    return parser.parse_args()


def load_experiments(mesh_args):
    experiments = {}
    for entry in mesh_args:
        if ":" not in entry:
            raise ValueError(f"Expected NAME:MESH_DIR, got: {entry}")
        name, mesh_dir = entry.split(":", 1)
        experiments[name] = mesh_dir
    return experiments


def compute_fixed_bbox(mesh_paths, margin):
    all_bounds = []
    for path in mesh_paths:
        m = trimesh.load(path, force="mesh")
        all_bounds.append(m.bounds)
    bb_min = np.min([b[0] for b in all_bounds], axis=0) - margin
    bb_max = np.max([b[1] for b in all_bounds], axis=0) + margin
    return bb_min, bb_max


def eval_mesh(mesh_path, ref_pts, n_samples):
    mesh = trimesh.load(mesh_path, force="mesh")
    surf_pts = trimesh.sample.sample_surface(mesh, min(n_samples, len(mesh.faces) * 3))[0]

    ref_o3d = o3d.geometry.PointCloud()
    ref_o3d.points = o3d.utility.Vector3dVector(ref_pts)
    tree_ref = o3d.geometry.KDTreeFlann(ref_o3d)

    surf_o3d = o3d.geometry.PointCloud()
    surf_o3d.points = o3d.utility.Vector3dVector(surf_pts)
    tree_surf = o3d.geometry.KDTreeFlann(surf_o3d)

    prec = np.array([np.sqrt(tree_ref.search_knn_vector_3d(pt, 1)[2][0]) for pt in surf_pts]) * 1000
    rec  = np.array([np.sqrt(tree_surf.search_knn_vector_3d(pt, 1)[2][0]) for pt in ref_pts]) * 1000

    return {
        "prec_mean": float(np.mean(prec)),
        "prec_p90":  float(np.percentile(prec, 90)),
        "rec_mean":  float(np.mean(rec)),
        "rec_p90":   float(np.percentile(rec, 90)),
        "faces":     len(mesh.faces),
    }


def main():
    args = parse_args()
    experiments = load_experiments(args.meshes)

    print(f"Loading reference point cloud: {args.ref}")
    pcd = o3d.io.read_point_cloud(args.ref)
    all_pts = np.asarray(pcd.points)
    print(f"  {len(all_pts):,} points\n")

    for obj_id in args.objects:
        mesh_paths = {name: os.path.join(d, f"{obj_id}.obj") for name, d in experiments.items()}

        missing = [n for n, p in mesh_paths.items() if not os.path.exists(p)]
        if missing:
            print(f"OBJECT {obj_id}: skipping — missing meshes for: {', '.join(missing)}")
            continue

        bb_min, bb_max = compute_fixed_bbox(list(mesh_paths.values()), args.bbox_margin)
        mask = np.all((all_pts >= bb_min) & (all_pts <= bb_max), axis=1)
        ref_pts = all_pts[mask]

        print(f"OBJECT {obj_id}  ({len(ref_pts):,} ref pts in fixed bbox)")
        print(f"  {'experiment':<20} {'prec_mean':>10} {'prec_p90':>9} {'rec_mean':>9} {'rec_p90':>9} {'faces':>7}")
        print(f"  {'':20} {'(mm)':>10} {'(mm)':>9} {'(mm)':>9} {'(mm)':>9}")

        for name, path in mesh_paths.items():
            r = eval_mesh(path, ref_pts, args.samples)
            print(f"  {name:<20} {r['prec_mean']:>10.2f} {r['prec_p90']:>9.2f} "
                  f"{r['rec_mean']:>9.2f} {r['rec_p90']:>9.2f} {r['faces']:>7}")
        print()


if __name__ == "__main__":
    main()
