#!/usr/bin/env python3
"""
CARLA 数据采集脚本（支持在线/离线两种模式）

功能:
1. 连接 CARLA 服务器，采集 RGB 图像 + 文本指令 + 轨迹
2. 将数据保存为 README 中定义的标准数据格式
3. 当本地未安装 CARLA 时，可使用离线模式快速生成伪数据
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import carla
except ImportError:  # pragma: no cover - 运行环境可能没有 CARLA
    carla = None


DEFAULT_COMMANDS = [
    "Follow the lane and maintain speed",
    "Turn left at the next intersection",
    "Turn right and merge into traffic",
    "Stop at the traffic light",
    "Overtake the vehicle ahead safely",
    "Change lane to the left",
    "Change lane to the right",
    "Slow down and prepare to stop",
    "Accelerate and maintain lane",
    "Navigate to the destination",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CARLA 数据采集")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA 主机地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA 端口")
    parser.add_argument("--town", type=str, default="Town03", help="CARLA 地图")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="数据集子目录")
    parser.add_argument("--episodes", type=int, default=2, help="采集 episode 数量")
    parser.add_argument("--frames-per-episode", type=int, default=600, help="每个 episode 采集的帧数")
    parser.add_argument("--future-steps", type=int, default=10, help="轨迹包含的未来时间步")
    parser.add_argument("--fps", type=float, default=20.0, help="仿真帧率")
    parser.add_argument("--output-dir", type=Path, default=Path("./datasets/carla"), help="输出根目录")
    parser.add_argument("--image-width", type=int, default=800, help="摄像头宽度")
    parser.add_argument("--image-height", type=int, default=600, help="摄像头高度")
    parser.add_argument("--camera-fov", type=float, default=90.0, help="摄像头视场角")
    parser.add_argument("--sensor-offset-x", type=float, default=1.6, help="摄像头前向偏移")
    parser.add_argument("--sensor-offset-z", type=float, default=1.5, help="摄像头高度")
    parser.add_argument("--offline", action="store_true", help="强制使用离线模式（无需 CARLA）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（离线模式）")
    return parser.parse_args()


class DatasetWriter:
    """负责落盘 images 与 annotations.json"""

    def __init__(self, root_dir: Path, split: str):
        self.split_dir = Path(root_dir) / split
        self.image_dir = self.split_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.annotations: Dict[str, Dict] = {}
        self.sample_index = 0

    def add_sample(
        self,
        image_rgb: np.ndarray,
        command: str,
        trajectory: Sequence[Sequence[float]],
        ego_position: Sequence[float],
    ):
        sample_id = f"{self.sample_index:06d}"
        image_path = self.image_dir / f"{sample_id}.png"
        Image.fromarray(image_rgb).save(image_path)
        self.annotations[sample_id] = {
            "image": f"images/{sample_id}.png",
            "command": command,
            "trajectory": [list(map(float, wp)) for wp in trajectory],
            "ego_position": [float(v) for v in ego_position],
        }
        self.sample_index += 1

    def finalize(self):
        annotation_path = self.split_dir / "annotations.json"
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2)
        print(f"✓ 写入标注文件: {annotation_path}")
        print(f"✓ 图像目录: {self.image_dir}")
        print(f"✓ 样本数量: {len(self.annotations)}")


class OfflineDataCollector:
    """当 CARLA 不可用时，快速生成结构一致的伪数据。"""

    def __init__(
        self,
        writer: DatasetWriter,
        frames_per_episode: int,
        episodes: int,
        future_steps: int,
        seed: int,
        commands: Sequence[str],
    ):
        self.writer = writer
        self.frames_per_episode = frames_per_episode
        self.episodes = episodes
        self.future_steps = future_steps
        self.rng = np.random.default_rng(seed)
        self.commands = list(commands)

    def run(self):
        print("⚠️  当前环境未检测到 CARLA，使用离线伪数据模式……")
        for episode in range(self.episodes):
            command = self.commands[episode % len(self.commands)]
            for frame in range(self.frames_per_episode):
                image = self._generate_image()
                trajectory = self._generate_trajectory()
                ego_pose = [0.0, 0.0, 0.0]
                self.writer.add_sample(image, command, trajectory, ego_pose)
            print(f"  - Episode {episode + 1}/{self.episodes} 完成（离线）")
        self.writer.finalize()

    def _generate_image(self) -> np.ndarray:
        img = self.rng.integers(0, 255, size=(600, 800, 3), dtype=np.uint8)
        return img

    def _generate_trajectory(self) -> List[List[float]]:
        t = np.linspace(0, 1, self.future_steps)
        x = t * self.rng.uniform(8.0, 14.0)
        y = np.sin(t * math.pi * self.rng.uniform(0.5, 1.5)) * self.rng.uniform(0.5, 2.0)
        return np.stack([x, y], axis=1).tolist()


class CarlaDataCollector:
    """在线模式：通过 CARLA 采集真实数据。"""

    def __init__(self, args: argparse.Namespace, writer: DatasetWriter, commands: Sequence[str]):
        if carla is None:
            raise RuntimeError("未找到 CARLA Python API，请安装 carla==0.9.15 或使用 --offline")

        self.args = args
        self.writer = writer
        self.commands = list(commands)

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.vehicle: Optional[carla.Actor] = None
        self.camera: Optional[carla.Sensor] = None
        self.original_settings: Optional[carla.WorldSettings] = None
        self.image_queue: "queue.Queue[carla.Image]" = queue.Queue()

    def __enter__(self):
        self._setup_world()
        self._spawn_vehicle()
        self._attach_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

    # --- Setup -----------------------------------------------------------------
    def _setup_world(self):
        print(f"连接 CARLA 服务器: {self.args.host}:{self.args.port} (Town={self.args.town})")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.args.town)

        settings = self.world.get_settings()
        self.original_settings = carla.WorldSettings(
            synchronous_mode=settings.synchronous_mode,
            fixed_delta_seconds=settings.fixed_delta_seconds,
            substepping=settings.substepping,
        )

        new_settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.args.fps,
            substepping=True,
        )
        self.world.apply_settings(new_settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        print("✓ 世界已切换为同步模式")

    def _spawn_vehicle(self):
        assert self.world is not None
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            raise RuntimeError("车辆生成失败，请检查地图是否有可用 spawn point")
        self.vehicle.set_autopilot(True)
        print(f"✓ Ego 车辆已生成: {self.vehicle.type_id} @ {spawn_point.location}")

    def _attach_camera(self):
        assert self.world is not None and self.vehicle is not None
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.args.image_width))
        camera_bp.set_attribute("image_size_y", str(self.args.image_height))
        camera_bp.set_attribute("fov", str(self.args.camera_fov))

        transform = carla.Transform(
            carla.Location(x=self.args.sensor_offset_x, z=self.args.sensor_offset_z),
            carla.Rotation(pitch=0.0),
        )
        self.camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.camera.listen(self.image_queue.put)
        print("✓ 摄像头已附加并开始监听帧数据")

    # --- Collection -------------------------------------------------------------
    def collect(self):
        assert self.world is not None
        total_episodes = self.args.episodes
        for episode_idx in range(total_episodes):
            instruction = self.commands[episode_idx % len(self.commands)]
            print(f"\n=== Episode {episode_idx + 1}/{total_episodes}: '{instruction}' ===")
            records = self._run_episode(instruction)
            self._flush_records(records, instruction)
        self.writer.finalize()

    def _run_episode(self, instruction: str) -> List[Dict]:
        records: List[Dict] = []
        assert self.world is not None and self.vehicle is not None

        warmup = self.args.future_steps
        total_frames = self.args.frames_per_episode + warmup
        print(f"开始采集，共 {total_frames} 帧 (含 warmup {warmup})")

        for frame_idx in range(total_frames):
            world_frame = self.world.tick()
            image = self._wait_for_image(world_frame)
            transform = self.vehicle.get_transform()
            location = transform.location
            yaw = math.radians(transform.rotation.yaw)

            image_rgb = self._carla_image_to_numpy(image)
            records.append(
                {
                    "frame": world_frame,
                    "image": image_rgb,
                    "location": np.array([location.x, location.y], dtype=np.float32),
                    "yaw": yaw,
                }
            )

            if frame_idx % 100 == 0:
                speed = self.vehicle.get_velocity()
                v = math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2) * 3.6
                print(f"  - Frame {frame_idx:04d} | yaw={math.degrees(yaw):6.2f}° | speed={v:5.1f} km/h")

        return records

    def _flush_records(self, records: List[Dict], instruction: str):
        future = self.args.future_steps
        print(f"回放 {len(records)} 帧生成轨迹样本 (future_steps={future})")
        for idx in range(len(records) - future):
            ego = records[idx]
            future_points = [records[idx + j]["location"] for j in range(future)]
            trajectory = [self._world_to_ego(ego, point) for point in future_points]
            ego_pose = [float(ego["location"][0]), float(ego["location"][1]), float(ego["yaw"])]
            self.writer.add_sample(
                image_rgb=ego["image"],
                command=instruction,
                trajectory=trajectory,
                ego_position=ego_pose,
            )

    # --- Helpers ----------------------------------------------------------------
    def _wait_for_image(self, expected_frame: int) -> "carla.Image":
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                image = self.image_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if image.frame == expected_frame:
                return image
        raise TimeoutError(f"等待摄像头像素帧 {expected_frame} 超时")

    @staticmethod
    def _carla_image_to_numpy(image: "carla.Image") -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1].copy()
        return rgb

    @staticmethod
    def _world_to_ego(ego: Dict, point: np.ndarray) -> List[float]:
        origin = ego["location"]
        yaw = ego["yaw"]
        delta = point - origin
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        x = delta[0] * cos_yaw - delta[1] * sin_yaw
        y = delta[0] * sin_yaw + delta[1] * cos_yaw
        return [float(x), float(y)]

    def _cleanup(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        if self.world is not None and self.original_settings is not None:
            self.world.apply_settings(self.original_settings)
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
        print("✓ 已清理 CARLA 资源")


def main():
    args = parse_args()
    commands = DEFAULT_COMMANDS
    writer = DatasetWriter(args.output_dir, args.split)

    online_mode = not args.offline and carla is not None

    if online_mode:
        collector = CarlaDataCollector(args, writer, commands)
        with collector:
            collector.collect()
    else:
        offline = OfflineDataCollector(
            writer=writer,
            frames_per_episode=args.frames_per_episode,
            episodes=args.episodes,
            future_steps=args.future_steps,
            seed=args.seed,
            commands=commands,
        )
        offline.run()


if __name__ == "__main__":
    sys.exit(main())
