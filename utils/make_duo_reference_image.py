"""Render the head-camera reference picture used in the VLM system prompt (axes drawn on the table, arms labelled).

    MUJOCO_GL=egl python utils/make_duo_reference_image.py src/vlagents/policies/assets/duo_frame_reference.png
"""

import sys

import numpy as np
import rcs
from PIL import Image, ImageDraw
from rcs.envs.base import ControlMode, RelativeTo


def project(point_shared: np.ndarray, shared2world, extrinsics: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    world = (shared2world * rcs.common.Pose(translation=point_shared)).translation()
    cam = extrinsics @ np.append(world, 1.0)
    pixel = intrinsics[:, :3] @ cam[:3]
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2])


def main(output: str):
    from duobench.tasks.transfer_cube import TransferCubeEnvConfig

    scene = TransferCubeEnvConfig()
    cfg = scene.config()
    cfg.headless = True
    cfg.control_mode = ControlMode.CARTESIAN_TRPY
    cfg.relative_to = RelativeTo.NONE
    cfg.camera_cfgs = {"head": cfg.camera_cfgs["head"]}
    cfg.camera_adds = {"head": cfg.camera_adds["head"]}
    env = scene.create_env(cfg)
    obs, _ = env.reset(seed=1)
    cameras = env.get_wrapper_attr("camera_set")
    extrinsics, intrinsics = cameras._extrinsics("head"), cameras._intrinsics("head")
    shared2world = cfg.root_frame_to_world * cfg.shared_base_frame_to_root_frame

    image = Image.fromarray(obs["frames"]["head"]["rgb"]["data"])
    draw = ImageDraw.Draw(image)
    origin = np.array([0.45, 0.0, -0.33])
    axes = {"+x (forward)": ([0.2, 0, 0], "red"), "+y (left)": ([0, 0.2, 0], "green"), "+z (up)": ([0, 0, 0.2], "blue")}
    o = project(origin, shared2world, extrinsics, intrinsics)
    for label, (direction, color) in axes.items():
        tip = project(origin + np.array(direction), shared2world, extrinsics, intrinsics)
        draw.line([o, tip], fill=color, width=6)
        draw.text((tip[0] + 8, tip[1] - 8), label, fill=color, font_size=28)
    draw.text((o[0] + 8, o[1] + 8), f"table point ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})", fill="black", font_size=24)
    for robot, label in (("left", "LEFT arm tool"), ("right", "RIGHT arm tool")):
        tcp = project(np.asarray(obs[robot]["tquat"][:3]), shared2world, extrinsics, intrinsics)
        draw.ellipse([tcp[0] - 10, tcp[1] - 10, tcp[0] + 10, tcp[1] + 10], outline="yellow", width=4)
        draw.text((tcp[0] - 80, tcp[1] - 45), label, fill="yellow", font_size=28)
    image.save(output)
    env.close()
    print("saved", output)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "duo_frame_reference.png")
