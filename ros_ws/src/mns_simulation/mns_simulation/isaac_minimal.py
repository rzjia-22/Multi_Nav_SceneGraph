"""Finite Isaac Lab smoke test for GPU, rendering, and PhysX bringup.

This intentionally has no ROS, robot, sensor, or project configuration
dependencies.  It is the Level 3 boundary between host/container preflight and
the real project runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--physics-rate", type=float, default=120.0)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


def main() -> int:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    if ARGS.steps < 2:
        raise ValueError("steps must be at least 2")
    if not math.isfinite(ARGS.physics_rate) or ARGS.physics_rate <= 0:
        raise ValueError("physics-rate must be positive and finite")

    simulation = SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / ARGS.physics_rate,
            render_interval=1,
            device=ARGS.device,
        )
    )
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=1200.0)
    light_cfg.func("/World/Light", light_cfg)
    cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/FallingCube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )

    simulation.reset()
    initial_height = float(cube.data.root_pos_w[0, 2])
    for _ in range(ARGS.steps):
        cube.write_data_to_sim()
        simulation.step(render=True)
        cube.update(simulation.get_physics_dt())
    final_height = float(cube.data.root_pos_w[0, 2])

    result = {
        "status": "PASS",
        "device": str(simulation.device),
        "physics_dt": simulation.get_physics_dt(),
        "steps": ARGS.steps,
        "initial_height_m": initial_height,
        "final_height_m": final_height,
    }
    if not math.isfinite(final_height) or not 0.05 <= final_height <= 0.25:
        result["status"] = "FAIL"
        result["reason"] = "rigid body did not settle on the ground"
    print("MNS_ISAAC_MINIMAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        # No outputs or Replicator workflow exist in this boundary smoke.  The
        # official immediate-close path avoids a reproduced 5.1 headless
        # shutdown hang after the success marker has already been emitted.
        APP.close(wait_for_replicator=False, skip_cleanup=True)
