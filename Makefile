SHELL := /bin/bash

.PHONY: lint test validate build robotics-image robotics-ml-image simulation-image gpu-preflight isaac-compatibility isaac-minimal model-source models test-models probe-go2-upstream phase1 phase1-diffusion uav-mapping phase2 phase1-synthetic phase1-synthetic-diffusion phase2-synthetic accept-isaac-sensors accept-go2-motion accept-phase1 accept-uav accept-phase2 inspect-hydra dataset-v0-scene-preview dataset-v0-episode-preview dataset-v0-validate dataset-v0-visualize

lint:
	python3 -m compileall -q ros_ws/src research_data tools tests
	python3 tools/validate_repository.py

test:
	PYTHONPATH=ros_ws/src/mns_core:ros_ws/src/mns_navigation:ros_ws/src/mns_motion:ros_ws/src/mns_multi_robot python3 -m pytest -q tests

validate: lint test

build:
	docker compose run --rm robotics-dev bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/hydra_ws/install/setup.bash && colcon build && colcon test && colcon test-result --verbose'

robotics-image:
	docker compose build robotics-dev

simulation-image:
	docker compose --profile simulation build simulation

gpu-preflight:
	bash tools/gpu_preflight.sh

isaac-compatibility: simulation-image
	docker compose --profile simulation run --rm --entrypoint bash simulation -lc \
		'/isaac-sim/isaac-sim.compatibility_check.sh --/app/quitAfter=20 --no-window'

isaac-minimal:
	docker compose --profile simulation run --rm simulation \
		/mns/containers/simulation/entrypoint.sh --minimal --headless

model-source:
	bash tools/verify_model_source.sh

robotics-ml-image: robotics-image model-source
	docker compose --profile ml build robotics-ml-dev

models:
	bash tools/fetch_model_assets.sh

test-models: model-source
	docker compose --profile ml run --rm robotics-ml-dev python3 /workspace/tools/smoke_diffusion.py

probe-go2-upstream: models
	docker compose --profile simulation run --rm --entrypoint /workspace/isaaclab/isaaclab.sh simulation \
		-p /mns/tools/go2_upstream_contract_probe.py --headless --steps 250 --profile training
	docker compose --profile simulation run --rm --entrypoint /workspace/isaaclab/isaaclab.sh simulation \
		-p /mns/tools/go2_upstream_contract_probe.py --headless --steps 750 --profile navigation

phase1:
	docker compose --profile simulation --profile robotics up simulation robotics-phase1

phase1-diffusion:
	docker compose --profile simulation --profile ml up simulation robotics-phase1-diffusion

uav-mapping:
	MNS_ROBOT_CONFIG=uav_mapping.yaml docker compose --profile simulation --profile robotics up simulation robotics-phase1

phase2:
	MNS_SCENARIO=phase2_team docker compose --profile simulation --profile robotics up simulation robotics-phase2

phase1-synthetic:
	docker compose run --rm robotics-dev ros2 launch mns_bringup phase1.launch.py simulation_mode:=synthetic navigator:=$${MNS_NAVIGATOR:-coverage} run_id:=$${MNS_RUN_ID:-synthetic-phase1}

phase1-synthetic-diffusion:
	docker compose --profile ml run --rm robotics-ml-dev ros2 launch mns_bringup phase1.launch.py simulation_mode:=synthetic navigator:=diffusion run_id:=$${MNS_RUN_ID:-synthetic-diffusion}

phase2-synthetic:
	docker compose run --rm robotics-dev ros2 launch mns_bringup phase2.launch.py simulation_mode:=synthetic navigator:=$${MNS_NAVIGATOR:-configured} run_id:=$${MNS_RUN_ID:-synthetic-phase2}

accept-phase1:
	docker compose run --rm robotics-dev python3 /workspace/tools/runtime_acceptance.py --robots go2_1 --duration $${MNS_ACCEPTANCE_DURATION:-12}

accept-uav:
	docker compose run --rm robotics-dev python3 /workspace/tools/runtime_acceptance.py --robots uav_1 --duration $${MNS_ACCEPTANCE_DURATION:-12}

accept-isaac-sensors:
	docker compose run --rm robotics-dev python3 /workspace/tools/isaac_sensor_acceptance.py \
		--robots $${MNS_ROBOTS:-go2_1} --duration $${MNS_ACCEPTANCE_DURATION:-30}

accept-go2-motion:
	docker compose run --rm robotics-dev python3 /workspace/tools/go2_motion_acceptance.py \
		--robot $${MNS_ROBOT:-go2_1} --wall-timeout $${MNS_ACCEPTANCE_TIMEOUT:-120}

accept-phase2:
	docker compose run --rm robotics-dev python3 /workspace/tools/runtime_acceptance.py --robots go2_1 go2_2 uav_1 uav_2 --duration $${MNS_ACCEPTANCE_DURATION:-12}

inspect-hydra:
	python3 tools/inspect_hydra_artifacts.py "$${MNS_HYDRA_DIR:?set MNS_HYDRA_DIR to a finalized Hydra output directory}"

dataset-v0-scene-preview:
	python3 -m research_data.cli generate-scene --scene-id train_scene_000
	python3 -m research_data.cli plan-preview

dataset-v0-episode-preview: dataset-v0-scene-preview
	docker compose --profile simulation run --rm --entrypoint /mns/containers/simulation/dataset_entrypoint.sh simulation \
		--scene /mns/research_scenes/dataset_v0/train_scene_000/scene.yaml \
		--plan /mns/artifacts/dataset_v0_preview/train_scene_000_episode_000/episode_plan.yaml \
		--output /mns/artifacts/dataset_v0_preview/train_scene_000_episode_000/episode.h5 \
		--headless --enable_cameras

dataset-v0-validate:
	docker compose --profile simulation run --rm --entrypoint /mns/containers/simulation/dataset_entrypoint.sh simulation \
		--tool validate --with-episode

dataset-v0-visualize:
	docker compose --profile simulation run --rm --entrypoint /mns/containers/simulation/dataset_entrypoint.sh simulation \
		--tool visualize
