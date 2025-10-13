#!/usr/bin/env python3
"""Project CARLA 3D bounding boxes into image space.

This script implements the workflow from the official CARLA "Bounding boxes"
Tutorial. It attaches an RGB camera to a hero vehicle, projects the bounding
boxes of surrounding actors to the camera plane, and optionally exports the
results for dataset creation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "numpy is required to run bbox.py. Install it with 'pip install numpy'."
    ) from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "OpenCV (cv2) is required to draw detections. Install it with 'pip install opencv-python'."
    ) from exc

import carla

# ---------------------------------------------------------------------------
# Utility helpers
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


@dataclass
class BoundingBox2D:
    actor_id: int
    label: str
    distance: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "actor_id": self.actor_id,
            "label": self.label,
            "distance": self.distance,
            "bbox": [self.x_min, self.y_min, self.x_max, self.y_max],
        }


@dataclass
class ActorProjection:
    bbox2d: BoundingBox2D
    vertices_image: List[np.ndarray]
    vertices_camera: List[np.ndarray]


class CarlaSyncMode:
    """Helper to keep world snapshots and sensor streams perfectly synchronized."""

    def __init__(self, world: carla.World, *sensors: carla.Sensor, fps: float | None = None):
        self.world = world
        self.sensors = sensors
        self.fps = fps
        self._queues: List[queue.Queue] = []
        self._settings: carla.WorldSettings | None = None
        self.frame: int | None = None

    def __enter__(self) -> CarlaSyncMode:
        original = self.world.get_settings()
        self._settings = original

        if self.fps and self.fps > 0.0:
            fixed_delta = 1.0 / self.fps
        else:
            fixed_delta = original.fixed_delta_seconds if original.fixed_delta_seconds else 0.05

        settings = carla.WorldSettings(
            synchronous_mode=True,
            no_rendering_mode=original.no_rendering_mode,
            fixed_delta_seconds=fixed_delta,
            substepping=original.substepping,
            max_substep_delta_time=original.max_substep_delta_time,
            max_substeps=original.max_substeps,
            max_culling_distance=original.max_culling_distance,
            deterministic_ragdolls=original.deterministic_ragdolls,
            tile_stream_distance=original.tile_stream_distance,
            actor_active_distance=original.actor_active_distance,
            spectator_as_ego=original.spectator_as_ego,
        )

        logging.debug("Enabling CarlaSyncMode (fixed_delta=%.3fs)", settings.fixed_delta_seconds)
        self.world.apply_settings(settings)

        self._queues.clear()

        def make_queue(register_event):
            q: queue.Queue = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

        return self

    def tick(self, timeout: float) -> List[object]:
        self.frame = self.world.tick()
        data = [self._retrieve(q, timeout) for q in self._queues]
        assert all(getattr(item, "frame", self.frame) == self.frame for item in data)
        return data

    def _retrieve(self, q: queue.Queue, timeout: float):
        while True:
            data = q.get(timeout=timeout)
            if getattr(data, "frame", None) == self.frame:
                return data

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        logging.debug("Restoring world settings after CarlaSyncMode")
        if self._settings is not None:
            self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.stop()


def spawn_hero(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    spawn_points: Sequence[carla.Transform],
    *,
    seed: int,
) -> carla.Vehicle:
    random.seed(seed)
    vehicle_blueprints = list(blueprint_library.filter("vehicle.*"))
    if not vehicle_blueprints:
        raise RuntimeError("No vehicle blueprints available.")

    spawn_candidates = list(spawn_points)
    random.shuffle(vehicle_blueprints)
    for blueprint in vehicle_blueprints:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            blueprint.set_attribute("color", random.choice(colors))
        random.shuffle(spawn_candidates)
        for transform in spawn_candidates:
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle:
                logging.info("Spawned hero vehicle %s", vehicle.type_id)
                return vehicle
    raise RuntimeError("Unable to spawn a hero vehicle. Try a different map or seed.")


def spawn_npcs(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    spawn_points: Sequence[carla.Transform],
    *,
    amount: int,
    traffic_manager: carla.TrafficManager,
    exclude_transforms: Iterable[carla.Transform],
    seed: int,
) -> List[carla.Vehicle]:
    random.seed(seed + 1)
    vehicle_blueprints = list(blueprint_library.filter("vehicle.*"))
    random.shuffle(vehicle_blueprints)

    excluded = list(exclude_transforms)
    available_spawns = list(spawn_points)
    random.shuffle(available_spawns)

    npcs: List[carla.Vehicle] = []
    for spawn_point in available_spawns:
        if any(spawn_point.location.distance(ex.location) < 1e-3 for ex in excluded):
            continue
        blueprint = random.choice(vehicle_blueprints)
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            blueprint.set_attribute("color", random.choice(colors))
        npc = world.try_spawn_actor(blueprint, spawn_point)
        if not npc:
            continue
        npc.set_autopilot(True, traffic_manager.get_port())
        npcs.append(npc)
        if len(npcs) >= amount:
            break
    logging.info("Spawned %d NPC vehicle(s)", len(npcs))
    return npcs


def attach_camera(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    vehicle: carla.Vehicle,
    *,
    width: int,
    height: int,
    fov: float,
    transform: carla.Transform,
) -> carla.Sensor:
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", str(fov))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    logging.info("Attached RGB camera to hero vehicle")
    return camera


BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (0, 4),
    (4, 5),
    (5, 1),
    (5, 7),
    (7, 6),
    (6, 4),
    (6, 2),
    (7, 3),
)


def build_projection_matrix(
    width: int,
    height: int,
    fov: float,
    *,
    flip: bool = False,
) -> np.ndarray:
    """Construct the intrinsic projection matrix used to map camera points to pixels."""
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    sign = -1.0 if flip else 1.0
    projection = np.zeros((3, 4), dtype=float)
    projection[0, 0] = sign * focal
    projection[1, 1] = sign * focal
    projection[0, 2] = width / 2.0
    projection[1, 2] = height / 2.0
    projection[2, 2] = 1.0
    return projection


def get_image_point(
    location: carla.Location,
    world_2_camera: np.ndarray,
    K_front: np.ndarray,
    K_back: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a world-space point to pixel coordinates and return the camera-space point."""
    point_camera = location_to_camera(location, world_2_camera)
    if abs(point_camera[2]) < 1e-6:
        point_camera[2] = 1e-6 if point_camera[2] >= 0 else -1e-6

    point_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
    projection = K_front if point_camera[2] >= 0 else K_back
    clip_space = projection @ point_h
    if abs(clip_space[2]) < 1e-6:
        clip_space[2] = 1e-6 if clip_space[2] >= 0 else -1e-6
    point = clip_space[:2] / clip_space[2]
    return point, point_camera


def location_to_camera(
    location: carla.Location,
    world_2_camera: np.ndarray,
) -> np.ndarray:
    """Transform a world-space location to the camera reference frame."""
    point = np.array([location.x, location.y, location.z, 1.0])
    point_camera = world_2_camera @ point
    # Convert from UE4 (X-forward, Y-right, Z-up) to standard (X-right, Y-down, Z-forward).
    return np.array([point_camera[1], -point_camera[2], point_camera[0]])


def point_in_canvas(point: np.ndarray, image_w: int, image_h: int) -> bool:
    return 0.0 <= point[0] < image_w and 0.0 <= point[1] < image_h


def compute_actor_projection(
    actor: carla.Actor,
    *,
    camera_transform: carla.Transform,
    world_2_camera: np.ndarray,
    K_front: np.ndarray,
    K_back: np.ndarray,
    image_w: int,
    image_h: int,
    max_distance: float,
    camera_location: carla.Location,
    ignore_actor_ids: set[int],
) -> ActorProjection | None:
    if actor.id in ignore_actor_ids:
        return None

    actor_transform = actor.get_transform()
    distance = actor_transform.location.distance(camera_location)
    if distance > max_distance:
        return None

    forward_vec = camera_transform.get_forward_vector()
    ray = actor_transform.location - camera_location
    if forward_vec.dot(ray) <= 0:
        return None

    bbox = actor.bounding_box
    vertices = bbox.get_world_vertices(actor_transform)
    vertices_camera: List[np.ndarray] = []
    vertices_image: List[np.ndarray] = []
    for vertex in vertices:
        image_point, camera_point = get_image_point(vertex, world_2_camera, K_front, K_back)
        vertices_camera.append(camera_point)
        vertices_image.append(image_point)

    if not any(vertex_cam[2] > 0 for vertex_cam in vertices_camera):
        return None

    xs = [point[0] for point in vertices_image]
    ys = [point[1] for point in vertices_image]

    x_min = max(min(xs), 0.0)
    y_min = max(min(ys), 0.0)
    x_max = min(max(xs), image_w - 1.0)
    y_max = min(max(ys), image_h - 1.0)

    if x_min >= x_max or y_min >= y_max:
        return None

    bbox2d = BoundingBox2D(
        actor_id=actor.id,
        label=actor.type_id,
        distance=distance,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )

    return ActorProjection(
        bbox2d=bbox2d,
        vertices_image=vertices_image,
        vertices_camera=vertices_camera,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D bounding boxes from a CARLA camera using the official tutorial workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=8000,
        help="Traffic Manager port used for NPC autopilot",
    )
    parser.add_argument("--map", default=None, help="Optional map to load (e.g. Town03)")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to capture")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Only persist every Nth frame while still collecting --frames outputs",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
        help="Maximum distance (m) for actors to be considered",
    )
    parser.add_argument(
        "--npcs",
        type=int,
        default=30,
        help="Number of background NPC vehicles to spawn",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Camera image width in pixels",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Camera image height in pixels",
    )
    parser.add_argument("--fov", type=float, default=90.0, help="Camera horizontal field of view")
    parser.add_argument(
        "--camera-z",
        type=float,
        default=2.2,
        help="Camera height relative to the vehicle roof (meters)",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.3,
        help="Camera forward offset from the vehicle origin (meters)",
    )
    parser.add_argument(
        "--hero-autopilot",
        action="store_true",
        help="Let the hero vehicle be controlled by the Traffic Manager",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory where images and annotations will be stored",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Convenience flag enabling --save-raw and --save-2d",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Persist the RGB frames captured by the camera",
    )
    parser.add_argument(
        "--save-2d",
        action="store_true",
        help="Save RGB frames with 2D bounding boxes",
    )
    parser.add_argument(
        "--save-3d",
        action="store_true",
        help="Save RGB frames with projected 3D bounding boxes",
    )
    parser.add_argument(
        "--save-voc",
        action="store_true",
        help="Export detections using the Pascal VOC XML format",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Dump a COCO-like JSON file with bounding boxes",
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
        help="Random seed used for blueprint sampling",
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
    map_spawn_points = list(world.get_map().get_spawn_points())
    if not map_spawn_points:
        raise RuntimeError("Selected map does not provide spawn points.")

    blueprint_library = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)

    hero = spawn_hero(world, blueprint_library, map_spawn_points, seed=args.seed)
    if args.hero_autopilot:
        hero.set_autopilot(True, traffic_manager.get_port())

    camera_transform = carla.Transform(
        carla.Location(x=args.camera_distance, z=args.camera_z),
        carla.Rotation(pitch=-15.0),
    )
    camera = attach_camera(
        world,
        blueprint_library,
        hero,
        width=args.camera_width,
        height=args.camera_height,
        fov=args.fov,
        transform=camera_transform,
    )

    npcs = spawn_npcs(
        world,
        blueprint_library,
        map_spawn_points,
        amount=args.npcs,
        traffic_manager=traffic_manager,
        exclude_transforms=[hero.get_transform()],
        seed=args.seed,
    )

    save_raw = args.save_raw or args.save_images or args.save_voc
    save_2d = args.save_2d or args.save_images
    save_3d = args.save_3d

    output_dir = args.output
    should_prepare_output = (
        save_raw or save_2d or save_3d or args.save_json or args.save_voc
    )
    if should_prepare_output:
        output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir: Path | None = output_dir / "raw" if save_raw else None
    detect_2d_dir: Path | None = output_dir / "detections_2d" if save_2d else None
    detect_3d_dir: Path | None = output_dir / "detections_3d" if save_3d else None

    for directory in (raw_dir, detect_2d_dir, detect_3d_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    Writer = None
    voc_dir: Path | None = None
    if args.save_voc:
        try:
            from pascal_voc_writer import Writer as PascalWriter  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "--save-voc requires 'pascal-voc-writer'. Install it with 'pip install pascal-voc-writer'."
            ) from exc
        Writer = PascalWriter
        voc_dir = output_dir / "voc"
        voc_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[dict] = []
    ignored_actor_ids = {hero.id}

    saved_frames = 0
    frame_counter = 0
    fps = 1.0 / args.fixed_delta if args.fixed_delta and args.fixed_delta > 0.0 else None

    try:
        with CarlaSyncMode(world, camera, fps=fps) as sync_mode:
            while saved_frames < args.frames:
                world_snapshot, image = sync_mode.tick(timeout=args.timeout)
                frame_counter += 1
                if frame_counter % args.frame_step != 0:
                    continue

                logging.info(
                    "[SYNC] world frame=%s, image frame=%s",
                    world_snapshot.frame,
                    image.frame,
                )

                camera_tf = camera.get_transform()
                world_2_camera = np.array(camera_tf.get_inverse_matrix())
                camera_location = camera_tf.location
                image_w = image.width
                image_h = image.height
                projection_front = build_projection_matrix(image_w, image_h, args.fov)
                projection_back = build_projection_matrix(image_w, image_h, args.fov, flip=True)

                frame_projections: List[ActorProjection] = []
                actors = world.get_actors().filter("*vehicle*")
                for actor in actors:
                    projection_result = compute_actor_projection(
                        actor,
                        camera_transform=camera_tf,
                        world_2_camera=world_2_camera,
                        K_front=projection_front,
                        K_back=projection_back,
                        image_w=image_w,
                        image_h=image_h,
                        max_distance=args.max_distance,
                        camera_location=camera_location,
                        ignore_actor_ids=ignored_actor_ids,
                    )
                    if projection_result:
                        frame_projections.append(projection_result)

                frame_boxes: List[BoundingBox2D] = [
                    projection.bbox2d for projection in frame_projections
                ]

                need_image_buffer = save_raw or save_2d or save_3d
                img_bgr = None
                if need_image_buffer:
                    img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
                    img_bgra = np.reshape(img_bgra, (image_h, image_w, 4))
                    img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

                raw_path: Path | None = None
                if save_raw and raw_dir and img_bgr is not None:
                    raw_path = raw_dir / f"{image.frame:06d}.png"
                    cv2.imwrite(str(raw_path), img_bgr)

                if save_2d and detect_2d_dir and img_bgr is not None:
                    img_2d = img_bgr.copy()
                    for box in frame_boxes:
                        pt1 = (int(round(box.x_min)), int(round(box.y_min)))
                        pt2 = (int(round(box.x_max)), int(round(box.y_max)))
                        cv2.rectangle(img_2d, pt1, pt2, (0, 0, 255), 2)
                    detect2d_path = detect_2d_dir / f"{image.frame:06d}.png"
                    cv2.imwrite(str(detect2d_path), img_2d)

                if save_3d and detect_3d_dir and img_bgr is not None:
                    img_3d = img_bgr.copy()
                    for projection in frame_projections:
                        for start, end in BOX_EDGES:
                            p_start = projection.vertices_image[start]
                            p_end = projection.vertices_image[end]
                            inside_start = point_in_canvas(p_start, image_w, image_h)
                            inside_end = point_in_canvas(p_end, image_w, image_h)
                            if not inside_start and not inside_end:
                                continue
                            if (
                                projection.vertices_camera[start][2] <= 0
                                and projection.vertices_camera[end][2] <= 0
                            ):
                                continue
                            pt1 = (int(round(p_start[0])), int(round(p_start[1])))
                            pt2 = (int(round(p_end[0])), int(round(p_end[1])))
                            cv2.line(img_3d, pt1, pt2, (0, 255, 0), 1)
                    detect3d_path = detect_3d_dir / f"{image.frame:06d}.png"
                    cv2.imwrite(str(detect3d_path), img_3d)

                if args.save_voc and Writer and voc_dir:
                    if raw_path is None and img_bgr is not None:
                        raw_path = output_dir / f"{image.frame:06d}.png"
                        cv2.imwrite(str(raw_path), img_bgr)
                    if raw_path is None:
                        logging.warning(
                            "Skipping VOC export for frame %s (image missing)", image.frame
                        )
                    else:
                        writer = Writer(str(raw_path), image_w, image_h)
                        for box in frame_boxes:
                            writer.addObject(
                                box.label,
                                int(round(box.x_min)),
                                int(round(box.y_min)),
                                int(round(box.x_max)),
                                int(round(box.y_max)),
                            )
                        voc_path = voc_dir / f"{image.frame:06d}.xml"
                        writer.save(str(voc_path))

                if args.save_json:
                    annotations.append(
                        {
                            "frame": int(image.frame),
                            "timestamp": float(image.timestamp),
                            "boxes": [box.as_dict() for box in frame_boxes],
                        }
                    )

                logging.debug(
                    "Frame %s captured %d bounding box(es)", image.frame, len(frame_boxes)
                )

                saved_frames += 1

    except KeyboardInterrupt:  # pragma: no cover - manual run
        logging.info("Interrupted by user")
    finally:
        camera.stop()
        logging.info("Destroying %d actors", 1 + len(npcs))
        client.apply_batch(
            [carla.command.DestroyActor(actor) for actor in [camera, hero, *npcs]]
        )

    if args.save_json and annotations:
        json_path = output_dir / "bounding_boxes.json"
        json_path.write_text(json.dumps({"frames": annotations}, indent=2))
        logging.info("Wrote annotations to %s", json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
