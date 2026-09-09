"""Safe simulator-local adapter for the audited Go2 RSL-RL actor.

The policy consumes the 48-value observation contract used by the upstream
Isaac Lab velocity task and returns 12 normalized joint-position offsets.
Navigation never imports this module and only publishes ``Twist``.
"""

from __future__ import annotations


ACTOR_INPUT_DIM = 48
ACTOR_OUTPUT_DIM = 12


def load_legacy_actor(checkpoint: str, device: str):
    """Reconstruct the actor MLP using tensor-only checkpoint loading."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Go2 checkpoint root must be a dictionary")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("Go2 checkpoint model_state_dict must be a dictionary")
    weights = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("actor.")
    }
    indices = sorted({
        int(key.split(".", 1)[0]) for key in weights if key.endswith(".weight")
    })
    if not indices:
        raise ValueError("checkpoint contains no legacy actor MLP")

    layers = []
    previous_output = None
    for number, index in enumerate(indices):
        weight_key, bias_key = f"{index}.weight", f"{index}.bias"
        if weight_key not in weights or bias_key not in weights:
            raise ValueError(f"actor layer {index} is incomplete")
        weight, bias = weights[weight_key], weights[bias_key]
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
            raise ValueError(f"actor layer {index} has invalid tensor shapes")
        if previous_output is not None and weight.shape[1] != previous_output:
            raise ValueError(f"actor layer {index} does not connect to its predecessor")
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.load_state_dict({"weight": weight, "bias": bias}, strict=True)
        layers.append(linear)
        previous_output = weight.shape[0]
        if number < len(indices) - 1:
            layers.append(torch.nn.ELU())

    actor = torch.nn.Sequential(*layers)
    if actor[0].in_features != ACTOR_INPUT_DIM or actor[-1].out_features != ACTOR_OUTPUT_DIM:
        raise ValueError(
            f"expected Go2 actor {ACTOR_INPUT_DIM}->{ACTOR_OUTPUT_DIM}, got "
            f"{actor[0].in_features}->{actor[-1].out_features}"
        )
    actor.requires_grad_(False)
    return actor.eval().to(device)


def velocity_observation(robot_data, velocity_commands, last_actions):
    """Build the upstream 48-value velocity-tracking observation in order."""
    import torch

    observation = torch.cat((
        robot_data.root_lin_vel_b,
        robot_data.root_ang_vel_b,
        robot_data.projected_gravity_b,
        velocity_commands,
        robot_data.joint_pos - robot_data.default_joint_pos,
        robot_data.joint_vel - robot_data.default_joint_vel,
        last_actions,
    ), dim=-1)
    if observation.shape[-1] != ACTOR_INPUT_DIM:
        raise ValueError(
            f"Go2 policy observation must have {ACTOR_INPUT_DIM} values, "
            f"got {observation.shape[-1]}"
        )
    return observation


class Go2VelocityPolicy:
    """Stateful 50 Hz actor wrapper that produces joint position targets."""

    def __init__(self, checkpoint: str, device: str, action_scale: float = 0.25) -> None:
        import torch

        if action_scale <= 0:
            raise ValueError("action_scale must be positive")
        self.torch = torch
        self.device = torch.device(device)
        self.action_scale = float(action_scale)
        self.actor = load_legacy_actor(checkpoint, device)
        self.last_actions = None

    def reset(self, robot_data) -> None:
        self.last_actions = self.torch.zeros(
            (robot_data.joint_pos.shape[0], ACTOR_OUTPUT_DIM),
            dtype=robot_data.joint_pos.dtype,
            device=self.device,
        )

    def joint_targets(self, robot_data, velocity_commands):
        if self.last_actions is None or self.last_actions.shape[0] != velocity_commands.shape[0]:
            self.reset(robot_data)
        commands = velocity_commands.to(device=self.device, dtype=robot_data.joint_pos.dtype)
        observation = velocity_observation(robot_data, commands, self.last_actions)
        with self.torch.inference_mode():
            actions = self.actor(observation)
        self.last_actions = actions
        return robot_data.default_joint_pos + self.action_scale * actions
