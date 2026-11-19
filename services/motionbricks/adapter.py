"""Loopback worker for one NVIDIA MotionBricks TensorRT rig.

Launch from the pinned NVIDIA runtime checkout using its own Python environment.
"""

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import newton
import numpy as np
import warp as wp

from motionbricks.io.utils import get_unitree_g1_model

# NVIDIA's headless example handles the BVH/CSV joint order and model-space conversion.
sys.path.insert(0, str(Path.cwd() / "docs" / "examples"))
from _rig import load_rig, sample_window


class Animator:
    def __init__(self, rig_id):
        self.name = "motionbricks-" + rig_id
        self.rig = load_rig("soma" if rig_id == "soma" else "unitree_g1")
        self.skeleton = self.rig.skeleton
        self.engine = self.rig.engine
        args = (self.skeleton,) if self.skeleton is not None else ()
        self.context = self.engine.create_input(*args)
        self.target = self.engine.create_input(*args)
        self.output = self.engine.create_output(*args)
        if self.skeleton is not None:
            self.joint_names = self.skeleton.joint_names
            self.bones = [[int(p), i] for i, p in enumerate(self.skeleton.parent_indices.numpy()) if p >= 0]
            self.world = wp.empty_like(self.skeleton.reference_local_transforms)
        else:
            self.joint_names = self.engine.contract.joint_names
            self.robot = get_unitree_g1_model(load_visuals=False)
            self.robot_state = self.robot.state()
            self.bones = [[int(p), int(c)] for p, c in zip(
                self.robot.joint_parent.numpy(), self.robot.joint_child.numpy()) if p >= 0 and c >= 0]

    def read_poses(self, values, label, minimum):
        poses = np.asarray(values)
        shape = (len(self.joint_names), 7) if self.skeleton is not None else (36,)
        if (poses.shape[1:] != shape or not minimum <= len(poses) <= 4
                or poses.dtype.kind not in "ifu"):
            raise ValueError(f"{label} requires {minimum}–4 numeric frames of shape {shape}")
        poses = poses.astype(np.float32)
        if not np.isfinite(poses).all():
            raise ValueError(f"{label} contains a non-finite coordinate")
        norms = np.linalg.norm(poses[..., 3:7], axis=-1)
        if not np.allclose(norms, 1.0, atol=0.01):
            raise ValueError(f"{label} requires unit XYZW quaternions")
        dtype = wp.transform if self.skeleton is not None else wp.float32
        return wp.array(poses, dtype=dtype, device=self.engine.device)

    def preview_positions(self):
        frames = []
        for pose_index in range(self.output.num_frames):
            pose = self.output.poses[pose_index]
            if self.skeleton is not None:
                self.skeleton.compute_world_transforms(pose, self.world, stream=self.engine.stream)
                world = self.world
            else:
                newton.eval_fk(self.robot, pose, self.robot.joint_qd, self.robot_state)
                world = self.robot_state.body_q
            frames.append(world.numpy()[:, :3].copy())
        return np.asarray(frames).round(5).tolist()

    def generate(self, body):
        if not isinstance(body, dict) or body.get("model") != self.name:
            raise ValueError(f"Select model {self.name}")
        with wp.ScopedStream(self.engine.stream):
            if set(body) == {"model", "example"} and body["example"] == "walk":
                context = wp.empty_like(self.context.poses)
                target = wp.empty_like(self.target.poses)
                sample_window(self.rig, 0.9, context)
                sample_window(self.rig, 1.8, target)
            elif set(body) == {"model", "context_poses", "target_poses"}:
                context = self.read_poses(body["context_poses"], "context_poses", 4)
                target = self.read_poses(body["target_poses"], "target_poses", 1)
            else:
                raise ValueError("Supply example=walk or context_poses and target_poses")
            self.context.update_from_poses(context, stream=self.engine.stream)
            self.target.update_from_poses(target, stream=self.engine.stream)
            self.engine.generate_animation(self.context, self.target, self.output)
            self.engine.synchronize()
            n = self.output.num_frames
            return {
                "model": self.name, "fps": 30, "num_frames": n, "up_axis": "Z",
                "pose_format": "local_xyz_xyzw" if self.skeleton is not None else "root_xyz_xyzw_joint_radians",
                "joint_names": list(self.joint_names), "frames": self.output.poses[:n].numpy().tolist(),
                "foot_contacts": self.output.foot_contacts[:n].numpy().tolist(),
                "input": {"context_poses": context.numpy().tolist(), "target_poses": target.numpy().tolist()},
                "preview": {"bones": self.bones, "positions": self.preview_positions()},
            }


class Handler(BaseHTTPRequestHandler):
    timeout = 15

    def reply(self, status, data):
        content = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"model": self.server.animator.name})
        else:
            self.reply(404, {"detail": "Unknown endpoint"})

    def do_POST(self):
        if self.path != "/v1/motions/generations":
            return self.reply(404, {"detail": "Unknown endpoint"})
        if self.headers.get_content_type() != "application/json" or self.headers.get("Origin"):
            return self.reply(415, {"detail": "Use the gateway to submit JSON"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 512 * 1024:
                return self.reply(413, {"detail": "Expected a JSON body of at most 512 KiB"})
            body = json.loads(self.rfile.read(size))
            result = self.server.animator.generate(body)
        except (ValueError, TypeError, OverflowError, RecursionError) as exc:
            return self.reply(400, {"detail": str(exc)})
        except Exception:
            traceback.print_exc()
            return self.reply(500, {"detail": "MotionBricks inference failed"})
        self.reply(200, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig", choices=("soma", "g1"), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    wp.init()
    wp.set_device("cuda:0")
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.animator = Animator(args.rig)
        try:
            server.serve_forever()
        finally:
            server.animator.engine.close()


if __name__ == "__main__":
    main()
