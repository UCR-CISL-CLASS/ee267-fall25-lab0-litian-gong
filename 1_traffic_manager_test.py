#!/usr/bin/env python3
"""Spawn reproducible traffic managed by CARLA's Traffic Manager.

This script follows the workflow demonstrated in CARLA's "Traffic manager"
Tutorial. It connects to a running simulator, optionally enables synchronous
mode, spawns vehicles, and hands control to the Traffic Manager (TM). It also
supports defining explicit waypoint paths that the TM will follow to reproduce
congestion scenarios.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Iterator, List, Sequence

import carla

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure_logging(verbosity: int) -> None:
    """Configure the logging level based on the desired verbosity."""
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
    world: carla.World, enable: bool, *, fixed_delta: float | None = None
) -> Iterator[None]:
    """Context manager to toggle synchronous simulation deterministically."""

    original = world.get_settings()
    if not enable:
        yield
        return

    # carla.WorldSettings lacks a copy constructor, so we rebuild a new instance explicitly.
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


@dataclass
class RouteSpec:
    """Definition of a path for the Traffic Manager to follow."""

    spawn_index: int
    waypoint_indices: List[int]


def parse_route_spec(spec: str) -> RouteSpec:
    """Parse a route specification string.

    Syntax: ``<spawn_index>:id0,id1``. Example: ``32:129,28,124``.
    """

    try:
        spawn_part, _, waypoints_part = spec.partition(":")
        spawn_idx = int(spawn_part.strip())
        waypoint_indices = [int(token) for token in waypoints_part.split(",") if token]
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise argparse.ArgumentTypeError(
            f"Route '{spec}' is not formatted as <spawn>:idx,idx"
        ) from exc

    if not waypoint_indices:
        raise argparse.ArgumentTypeError(
            f"Route '{spec}' must contain at least one waypoint index"
        )

    return RouteSpec(spawn_index=spawn_idx, waypoint_indices=waypoint_indices)


def choose_vehicle_blueprint_ids(blueprint_library: carla.BlueprintLibrary) -> List[str]:
    """Return blueprint identifiers for a diverse set of vehicles."""
    preferred_models = {
        "dodge",
        "audi",
        "model3",
        "mini",
        "mustang",
        "lincoln",
        "prius",
        "nissan",
        "crown",
        "impala",
    }

    selected: List[str] = []
    for blueprint in blueprint_library.filter("vehicle.*"):
        if any(model in blueprint.id for model in preferred_models):
            selected.append(blueprint.id)

    if not selected:
        selected = [bp.id for bp in blueprint_library.filter("vehicle.*")]
    return selected


def new_vehicle_blueprint(
    blueprint_library: carla.BlueprintLibrary, blueprint_id: str
) -> carla.ActorBlueprint:
    """Retrieve a fresh blueprint instance to avoid side-effects between spawns."""
    return blueprint_library.find(blueprint_id)


def randomise_vehicle_attributes(blueprint: carla.ActorBlueprint) -> None:
    """Randomise optional vehicle blueprint attributes."""
    if blueprint.has_attribute("color"):
        colors = blueprint.get_attribute("color").recommended_values
        blueprint.set_attribute("color", random.choice(colors))
    if blueprint.has_attribute("driver_id"):
        drivers = blueprint.get_attribute("driver_id").recommended_values
        blueprint.set_attribute("driver_id", random.choice(drivers))
    if blueprint.has_attribute("is_invincible"):
        blueprint.set_attribute("is_invincible", "false")


def spawn_vehicle(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    spawn_transform: carla.Transform,
    traffic_manager_port: int,
    *,
    hero: bool = False,
) -> carla.Vehicle | None:
    """Try to spawn a single vehicle and hand control to the TM."""
    if hero:
        blueprint.set_attribute("role_name", "hero")
    vehicle = world.try_spawn_actor(blueprint, spawn_transform)
    if vehicle:
        vehicle.set_autopilot(True, traffic_manager_port)
    return vehicle


def configure_traffic_manager(tm: carla.TrafficManager, args: argparse.Namespace) -> None:
    """Apply global TM configuration from CLI arguments."""
    tm.set_synchronous_mode(args.sync)
    tm.set_random_device_seed(args.seed)
    tm.set_global_distance_to_leading_vehicle(args.distance)
    tm.global_percentage_speed_difference(args.speed_offset)
    tm.set_hybrid_physics_mode(args.hybrid_physics)
    tm.set_respawn_dormant_vehicles(True)


def apply_per_vehicle_settings(
    tm: carla.TrafficManager,
    vehicle: carla.Vehicle,
    args: argparse.Namespace,
) -> None:
    """Configure TM behaviour for a single vehicle using CLI options."""
    if args.no_lane_change:
        tm.auto_lane_change(vehicle, False)
        tm.random_left_lanechange_percentage(vehicle, 0)
        tm.random_right_lanechange_percentage(vehicle, 0)
    if args.ignore_lights is not None:
        lower, upper = args.ignore_lights
        tm.ignore_lights_percentage(vehicle, random.randint(lower, upper))


def build_route(
    spec: RouteSpec,
    spawn_points: Sequence[carla.Transform],
) -> tuple[carla.Transform, List[carla.Location]]:
    """Create transform and waypoint list from indices."""
    if spec.spawn_index >= len(spawn_points) or spec.spawn_index < 0:
        raise ValueError(
            f"Spawn index {spec.spawn_index} is out of bounds for map with {len(spawn_points)} spawn points"
        )

    route_spawn = spawn_points[spec.spawn_index]
    waypoint_locations: List[carla.Location] = []
    for waypoint_index in spec.waypoint_indices:
        if waypoint_index >= len(spawn_points) or waypoint_index < 0:
            raise ValueError(
                f"Waypoint index {waypoint_index} out of bounds ({len(spawn_points)} spawn points)"
            )
        waypoint_locations.append(spawn_points[waypoint_index].location)
    return route_spawn, waypoint_locations


def draw_spawn_annotations(
    world: carla.World,
    spawn_points: Sequence[carla.Transform],
    *,
    lifetime: float = 20.0,
) -> None:
    """Visualise spawn indices in the world for quick debugging."""
    for index, transform in enumerate(spawn_points):
        world.debug.draw_string(
            transform.location,
            str(index),
            life_time=lifetime,
            color=carla.Color(255, 255, 0),
        )


def set_spectator_follow(spectator: carla.Actor, target: carla.Actor) -> None:
    """Place the spectator above the tracked actor with a trailing offset."""
    transform = target.get_transform()
    location = transform.location
    rotation = transform.rotation

    distance = 10.0
    yaw_radians = math.radians(rotation.yaw + 180.0)

    spectator.set_transform(
        carla.Transform(
            carla.Location(
                x=location.x + distance * math.cos(yaw_radians),
                y=location.y + distance * math.sin(yaw_radians),
                z=location.z + 6.0,
            ),
            carla.Rotation(pitch=-20.0, yaw=rotation.yaw, roll=0.0),
        )
    )


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment with CARLA's Traffic Manager using reproducible settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=8000,
        help="Port of the Traffic Manager instance",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for TM and sampling")
    parser.add_argument(
        "--vehicles",
        type=int,
        default=50,
        help="Maximum number of vehicles to spawn (capped by available spawn points)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Simulation time to keep the scenario alive (seconds)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Enable synchronous mode for deterministic steps",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=None,
        help="Fixed delta time when running synchronously",
    )
    parser.add_argument(
        "--map",
        default=None,
        help="Optional town to load before spawning (e.g. Town03)",
    )
    parser.add_argument(
        "--draw-spawn-points",
        action="store_true",
        help="Annotate all spawn points to match tutorial visuals",
    )
    parser.add_argument(
        "--routes",
        type=parse_route_spec,
        nargs="*",
        help="Optional route definitions that vehicles will follow (format: spawn:idx,idx)",
    )
    parser.add_argument(
        "--speed-offset",
        type=float,
        default=0.0,
        help="Speed offset percentage applied by the TM (negative slows down)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=2.5,
        help="Global TM minimum distance to the leading vehicle (meters)",
    )
    parser.add_argument(
        "--ignore-lights",
        type=str,
        default=None,
        help="Range 'min,max' defining probability (0-100) of ignoring traffic lights",
    )
    parser.add_argument(
        "--no-lane-change",
        action="store_true",
        help="Disable automatic lane changes in the TM",
    )
    parser.add_argument(
        "--hybrid-physics",
        action="store_true",
        help="Enable hybrid physics (recommended for large towns)",
    )
    parser.add_argument(
        "--spectator-follow",
        action="store_true",
        help="Keep the spectator camera following the first spawned vehicle",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Logging verbosity (0=warnings, 2=debug)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Client connection timeout",
    )
    args = parser.parse_args(argv)

    if args.ignore_lights is not None:
        try:
            tokens = [int(token) for token in args.ignore_lights.split(",")]
            if len(tokens) != 2:
                raise ValueError
            if tokens[0] < 0 or tokens[1] > 100 or tokens[0] > tokens[1]:
                raise ValueError
            args.ignore_lights = (tokens[0], tokens[1])
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise argparse.ArgumentTypeError(
                "--ignore-lights must be formatted as min,max with 0 <= min <= max <= 100"
            ) from exc

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbosity)
    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world() if args.map is None else client.load_world(args.map)
    spectator = world.get_spectator()
    blueprint_library = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)

    configure_traffic_manager(traffic_manager, args)

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("The selected map has no spawn points available.")

    if args.draw_spawn_points:
        draw_spawn_annotations(world, spawn_points)

    # Prepare routes if provided, otherwise fall back to the example from docs.
    routes = args.routes or [
        parse_route_spec("32:129,28,124,33,97,119,58,154,147"),
        parse_route_spec("149:21,76,38,34,90,3"),
    ]

    blueprint_ids = choose_vehicle_blueprint_ids(blueprint_library)
    vehicles: List[carla.Vehicle] = []
    used_spawn_indices = set()

    # Spawn vehicles for each route first to ensure traffic on these paths.
    for idx, route_spec in enumerate(routes):
        spawn_transform, waypoint_locations = build_route(route_spec, spawn_points)
        random.shuffle(blueprint_ids)
        spawned: carla.Vehicle | None = None
        for blueprint_id in blueprint_ids:
            blueprint = new_vehicle_blueprint(blueprint_library, blueprint_id)
            randomise_vehicle_attributes(blueprint)
            spawned = spawn_vehicle(
                world,
                blueprint,
                spawn_transform,
                args.traffic_manager_port,
                hero=idx == 0,
            )
            if spawned:
                break
        if not spawned:
            logging.warning(
                "Failed to spawn route vehicle at spawn index %d", route_spec.spawn_index
            )
            continue
        used_spawn_indices.add(route_spec.spawn_index)
        apply_per_vehicle_settings(traffic_manager, spawned, args)
        traffic_manager.set_path(spawned, waypoint_locations)
        vehicles.append(spawned)
        logging.info(
            "Spawned route vehicle %s with %d waypoints",
            spawned.type_id,
            len(waypoint_locations),
        )

    # Spawn remaining random traffic up to the requested limit.
    available_spawn_points = [
        sp for idx, sp in enumerate(spawn_points) if idx not in used_spawn_indices
    ]
    random.shuffle(available_spawn_points)
    capacity = min(args.vehicles - len(vehicles), len(available_spawn_points))
    for spawn_point in available_spawn_points[:capacity]:
        random.shuffle(blueprint_ids)
        spawned = None
        for blueprint_id in blueprint_ids:
            blueprint = new_vehicle_blueprint(blueprint_library, blueprint_id)
            randomise_vehicle_attributes(blueprint)
            spawned = spawn_vehicle(
                world,
                blueprint,
                spawn_point,
                args.traffic_manager_port,
            )
            if spawned:
                break
        if not spawned:
            continue
        apply_per_vehicle_settings(traffic_manager, spawned, args)
        vehicles.append(spawned)

    if not vehicles:
        raise RuntimeError("Unable to spawn any vehicles. Try reducing --vehicles or change map.")

    logging.info("Handed %d vehicle(s) to the Traffic Manager", len(vehicles))

    start_time = time.time()
    try:
        with synchronous_mode(world, args.sync, fixed_delta=args.fixed_delta):
            while True:
                if args.sync:
                    world.tick()
                else:
                    world.wait_for_tick()

                if args.spectator_follow and vehicles:
                    set_spectator_follow(spectator, vehicles[0])

                elapsed = time.time() - start_time
                if elapsed >= args.duration:
                    break
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        logging.info("Interrupted by user")
    finally:
        logging.info("Destroying %d vehicle(s)", len(vehicles))
        client.apply_batch([carla.command.DestroyActor(vehicle) for vehicle in vehicles])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
