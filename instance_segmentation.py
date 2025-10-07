#!/usr/bin/env python3
"""Capture instance segmentation images with CARLA.

The code mirrors the workflow presented in CARLA's "Instance segmentation sensor"
Tutorial: it positions an instance segmentation camera at a fixed viewpoint,
spawns surrounding traffic, captures a sequence of frames, and writes the
results to disk.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import queue
import random
import sys
from pathlib import Path
from typing import Iterator, List, Sequence

import carla

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


@contextlib.contextmanager
def synchronous_mode(
    world: carla.World,
    enable: bool,
    *,
    fixed_delta: float | None = None,
) -> Iterator[None]:
    """Context manager to temporarily enable synchronous simulation."""

    original = world.get_settings()
    if not enable:
        yield
        return

    # carla.WorldSettings 没有复制构造函数，需要手动重建设置。
    target_fixed_delta = (
        fixed_delta
        if fixed_delta is not None
        else (original.fixed_delta_seconds if original.fixed_delta_seconds else 0.05)
    )

    settings = carla.WorldSettings(
        synchronous_mode=True,
        no_rendering_mode=original.no_rendering_mode,
        fixed_delta_seconds=target_fixed_delta,
        substepping=original.substepping,
        max_substep_delta_time=original.max_substep_delta_time,
        max_substeps=original.max_substeps,
        max_culling_distance=original.max_culling_distance,
        deterministic_ragdolls=original.deterministic_ragdolls,
        tile_stream_distance=original.tile_stream_distance,
        actor_active_distance=original.actor_active_distance,
        spectator_as_ego=original.spectator_as_ego,
    )

    logging.debug("Enabling synchronous mode (fixed_delta=%.3fs)", target_fixed_delta)
    world.apply_settings(settings)
    try:
        yield
    finally:
        logging.debug("Restoring original world settings")
        world.apply_settings(original)


def spawn_vehicle_cloud(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    spawn_points: Sequence[carla.Transform],
    *,
    center: carla.Location,
    radius: float,
    limit: int,
    traffic_manager: carla.TrafficManager | None,
    seed: int,
) -> List[carla.Vehicle]:
    random.seed(seed)
    blueprints = list(blueprint_library.filter("vehicle.*"))
    if not blueprints:
        return []

    vehicles: List[carla.Vehicle] = []
    sq_radius = radius * radius
    random.shuffle(blueprints)

    for spawn_point in spawn_points:
        if len(vehicles) >= limit:
            break
        dx = spawn_point.location.x - center.x
        dy = spawn_point.location.y - center.y
        if dx * dx + dy * dy > sq_radius:
            continue
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            blueprint.set_attribute("color", random.choice(colors))
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if not vehicle:
            continue
        if traffic_manager is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
        vehicles.append(vehicle)

    logging.info("Spawned %d background vehicle(s)", len(vehicles))
    return vehicles


def attach_instance_camera(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    transform: carla.Transform,
) -> carla.Sensor:
    camera_bp = blueprint_library.find("sensor.camera.instance_segmentation")
    camera_bp.set_attribute("image_size_x", "1280")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    sensor = world.spawn_actor(camera_bp, transform)
    logging.info("Spawned instance segmentation sensor")
    return sensor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture instance segmentation images from CARLA following the official tutorial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument("--map", default=None, help="Optional map to load (e.g. Town03)")
    parser.add_argument(
        "--camera-location",
        type=float,
        nargs=3,
        default=(-46.0, 152.0, 18.0),
        metavar=("X", "Y", "Z"),
        help="World-space location for the camera as used in the tutorial",
    )
    parser.add_argument(
        "--camera-rotation",
        type=float,
        nargs=3,
        default=(-21.0, -93.4, 0.0),
        metavar=("PITCH", "YAW", "ROLL"),
        help="Orientation of the instance segmentation camera",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=80.0,
        help="Radius (m) used when spawning background vehicles",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=50,
        help="Maximum number of background vehicles to try spawning",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of segmentation frames to capture",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Only persist every Nth frame while still collecting --frames outputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("instance_output"),
        help="Directory to store the captured frames",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run the world in synchronous mode for deterministic captures",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
        help="Fixed delta time when running synchronously",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when sampling vehicle blueprints",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Logging verbosity",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Client timeout in seconds",
    )
    args = parser.parse_args(argv)
    if args.frame_step <= 0:
        parser.error("--frame-step must be a positive integer")
    return args


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbosity)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world() if args.map is None else client.load_world(args.map)
    blueprint_library = world.get_blueprint_library()

    camera_location = carla.Location(*args.camera_location)
    camera_rotation = carla.Rotation(*args.camera_rotation)
    camera_transform = carla.Transform(camera_location, camera_rotation)

    spectator = world.get_spectator()
    spectator.set_transform(camera_transform)

    spawn_points = list(world.get_map().get_spawn_points())

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(args.sync)
    traffic_manager.set_random_device_seed(args.seed)

    background_vehicles = spawn_vehicle_cloud(
        world,
        blueprint_library,
        spawn_points,
        center=camera_location,
        radius=args.radius,
        limit=args.vehicles,
        traffic_manager=traffic_manager,
        seed=args.seed,
    )

    sensor = attach_instance_camera(world, blueprint_library, camera_transform)

    sensor_queue: queue.Queue[carla.Image] = queue.Queue()
    sensor.listen(sensor_queue.put)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = 0
    frame_counter = 0

    try:
        with synchronous_mode(world, args.sync, fixed_delta=args.fixed_delta):
            while saved_frames < args.frames:
                if args.sync:
                    world.tick()
                else:
                    world.wait_for_tick()

                image = sensor_queue.get(timeout=args.timeout)
                frame_counter += 1
                should_save = frame_counter % args.frame_step == 0

                if not should_save:
                    logging.debug(
                        "Skipping frame %s (frame-step=%d)",
                        image.frame,
                        args.frame_step,
                    )
                    continue

                image_path = output_dir / f"instance_{image.frame:06d}.png"
                image.save_to_disk(str(image_path))
                logging.info("Saved %s", image_path)
                saved_frames += 1
    except KeyboardInterrupt:  # pragma: no cover - manual run
        logging.info("Interrupted by user")
    finally:
        sensor.stop()
        actors_to_destroy = [sensor, *background_vehicles]
        logging.info("Destroying %d actors", len(actors_to_destroy))
        client.apply_batch([carla.command.DestroyActor(actor) for actor in actors_to_destroy])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
