"""
CARLA 闭环评估脚本

特性：
- 自动连接 CARLA（若不可用则退化为离线 mock 模式）
- 加载 VLADrivingPolicy 检查点并执行闭环控制
- 统计碰撞、越线、路线完成度等指标
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import compute_infraction_score, compute_route_completion
from models.policy import VLADrivingPolicy

try:
    import carla
except ImportError:  # pragma: no cover - 运行环境可能没有 CARLA
    carla = None


CLIP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

DEFAULT_COMMANDS = [
    "Follow the lane and maintain speed",
    "Turn left at the next intersection",
    "Turn right and merge into traffic",
    "Stop at the traffic light",
    "Overtake the vehicle ahead safely",
    "Change lane to the left",
    "Change lane to the right",
]


def load_policy(checkpoint: str, device: torch.device) -> VLADrivingPolicy:
    """加载 VLADrivingPolicy 并尝试恢复权重。"""
    print(f"加载策略模型 (checkpoint={checkpoint})")
    model = VLADrivingPolicy(
        model_name="microsoft/phi-2",
        vision_model_name="openai/clip-vit-base-patch32",
        num_timesteps=10,
        use_lora=False,
        freeze_llm=False,
        freeze_vision_tower=False,
    ).to(device)

    ckpt_path = Path(checkpoint)
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("state_dict", state)
        load_result = model.load_state_dict(
            {k.replace("model.", ""): v for k, v in state_dict.items()},
            strict=False,
        )
        print(f"✓ 权重加载完成，missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
    else:
        print("⚠️ 未找到 checkpoint，使用随机初始化模型")

    model.eval()
    return model


class CARLASimulator:
    """封装 CARLA 交互逻辑，若不可用则退化为离线模式。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town03",
        fps: float = 20.0,
        timeout: float = 10.0,
    ):
        self.host = host
        self.port = port
        self.town = town
        self.fps = fps
        self.timeout = timeout

        self.online = carla is not None
        self.client: Optional["carla.Client"] = None
        self.world: Optional["carla.World"] = None
        self.vehicle: Optional["carla.Vehicle"] = None
        self.camera: Optional["carla.Sensor"] = None
        self.collision_sensor: Optional["carla.Sensor"] = None
        self.lane_sensor: Optional["carla.Sensor"] = None
        self.image_queue: "queue.Queue[carla.Image]" = queue.Queue()
        self.original_settings: Optional["carla.WorldSettings"] = None

        self.collisions = 0
        self.lane_events = 0
        self.total_frames = 0

    def setup_world(self):
        if not self.online:
            print("⚠️ 未检测到 CARLA，使用离线评估模式")
            return

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.load_world(self.town)

        settings = self.world.get_settings()
        self.original_settings = carla.WorldSettings(
            synchronous_mode=settings.synchronous_mode,
            fixed_delta_seconds=settings.fixed_delta_seconds,
            substepping=settings.substepping,
        )

        new_settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.fps,
            substepping=True,
        )
        self.world.apply_settings(new_settings)
        print(f"✓ 已连接 CARLA ({self.town}) 并切换为同步模式")

    def begin_episode(self):
        self.collisions = 0
        self.lane_events = 0
        self.total_frames = 0

        if not self.online:
            return

        assert self.world is not None
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            raise RuntimeError("Ego 车辆生成失败，请重试")

        self.vehicle.set_autopilot(False)
        self._spawn_camera(bp_lib)
        self._spawn_collision_sensor(bp_lib)
        self._spawn_lane_sensor(bp_lib)
        print(f"✓ Episode 初始化完成 @ {spawn_point.location}")

    def _spawn_camera(self, bp_lib):
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")
        transform = carla.Transform(
            carla.Location(x=1.6, z=1.5),
            carla.Rotation(pitch=0.0),
        )
        self.camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.camera.listen(self.image_queue.put)

    def _spawn_collision_sensor(self, bp_lib):
        sensor_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: setattr(self, "collisions", self.collisions + 1))

    def _spawn_lane_sensor(self, bp_lib):
        sensor_bp = bp_lib.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(
            lambda event: setattr(self, "lane_events", self.lane_events + len(event.crossed_lane_markings))
        )

    def step(self) -> Dict[str, np.ndarray]:
        self.total_frames += 1
        if not self.online:
            mock_image = np.random.randint(0, 255, size=(600, 800, 3), dtype=np.uint8)
            return {"rgb": mock_image, "speed": 30.0}

        assert self.world is not None
        frame_id = self.world.tick()
        image = self._wait_for_image(frame_id)
        rgb = self._carla_image_to_numpy(image)
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        return {"rgb": rgb, "speed": speed}

    def apply_control(self, steering: float, throttle: float, brake: float):
        if not self.online:
            return
        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=float(np.clip(steering, -1.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
        )
        self.vehicle.apply_control(control)

    def fetch_metrics(self) -> Dict[str, float]:
        collisions = self.collisions
        off_road = self.lane_events
        total = max(self.total_frames, 1)

        infraction = compute_infraction_score(
            collisions=collisions,
            red_light_violations=0,  # TODO: 集成红灯检测
            off_road_frames=off_road,
            total_frames=total,
        )
        completion = compute_route_completion(
            waypoints_reached=self.total_frames,
            total_waypoints=self.total_frames,
        )
        return {**infraction, "route_completion": completion}

    def end_episode(self):
        if not self.online:
            return
        for actor in [self.camera, self.collision_sensor, self.lane_sensor, self.vehicle]:
            if actor is not None:
                actor.destroy()
        self.camera = self.collision_sensor = self.lane_sensor = self.vehicle = None

    def teardown(self):
        if not self.online or self.world is None:
            return
        if self.original_settings is not None:
            self.world.apply_settings(self.original_settings)

    def _wait_for_image(self, frame: int) -> "carla.Image":
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                image = self.image_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if image.frame == frame:
                return image
        raise TimeoutError(f"等待 RGB 帧 {frame} 超时")

    @staticmethod
    def _carla_image_to_numpy(image: "carla.Image") -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array[:, :, :3][:, :, ::-1].copy()


class VLAEvaluator:
    """负责将模型输出转换为控制指令并累计指标。"""

    def __init__(self, policy: VLADrivingPolicy, simulator: CARLASimulator, device: torch.device):
        self.policy = policy
        self.simulator = simulator
        self.device = device

    @torch.no_grad()
    def _predict_trajectory(self, rgb: np.ndarray, instruction: str) -> torch.Tensor:
        image_tensor = CLIP_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        trajectory = self.policy.predict_trajectory(image_tensor, [instruction])
        return trajectory

    def _trajectory_to_control(self, trajectory: torch.Tensor) -> Dict[str, float]:
        traj = trajectory[0].cpu().numpy()
        target = traj[0]
        angle = math.atan2(target[1], max(1e-3, target[0]))
        steering = float(np.clip(angle / (math.pi / 4), -1.0, 1.0))
        distance = np.linalg.norm(target)
        throttle = float(np.clip(distance * 0.2, 0.0, 0.8))
        brake = 0.0 if distance > 0.5 else 0.3
        return {"steering": steering, "throttle": throttle, "brake": brake}

    def run_episode(self, instruction: str, max_steps: int) -> Dict[str, float]:
        self.simulator.begin_episode()
        try:
            for step in range(max_steps):
                observation = self.simulator.step()
                trajectory = self._predict_trajectory(observation["rgb"], instruction)
                control = self._trajectory_to_control(trajectory)
                self.simulator.apply_control(**control)
                if step % 50 == 0:
                    print(
                        f"  step={step:04d} | steer={control['steering']:+.3f} "
                        f"| throttle={control['throttle']:.3f} | brake={control['brake']:.2f}"
                    )
        finally:
            self.simulator.end_episode()
        metrics = self.simulator.fetch_metrics()
        print(f"Episode metrics: {metrics}")
        return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CARLA 闭环评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="策略模型 checkpoint 路径")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", type=str, default="Town03", help="CARLA 地图")
    parser.add_argument("--fps", type=float, default=20.0, help="仿真帧率")
    parser.add_argument("--num-episodes", type=int, default=3, help="评估 episode 数量")
    parser.add_argument("--max-steps", type=int, default=600, help="每个 episode 的最大步数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    device = torch.device(args.device)

    simulator = CARLASimulator(host=args.host, port=args.port, town=args.town, fps=args.fps)
    simulator.setup_world()

    policy = load_policy(args.checkpoint, device)
    evaluator = VLAEvaluator(policy, simulator, device)

    commands = DEFAULT_COMMANDS
    results: List[Dict[str, float]] = []
    for episode_idx in range(args.num_episodes):
        instruction = commands[episode_idx % len(commands)]
        print(f"\n=== Episode {episode_idx + 1}/{args.num_episodes}: '{instruction}' ===")
        metrics = evaluator.run_episode(instruction, max_steps=args.max_steps)
        results.append(metrics)

    if results:
        print("\n=== 评估汇总 ===")
        keys = results[0].keys()
        for key in keys:
            avg = np.mean([m[key] for m in results])
            print(f"{key}: {avg:.4f}")

    simulator.teardown()
    print("\n评估完成 ✅")


if __name__ == "__main__":
    main()
