SHELL := /bin/bash

.PHONY: lint test validate build robotics-image robotics-ml-image model-source models test-models phase1 phase1-diffusion phase2 phase1-synthetic phase1-synthetic-diffusion phase2-synthetic accept-phase1 accept-phase2

lint:
	python3 -m compileall -q ros_ws/src tools tests
	python3 tools/validate_repository.py

test:
	PYTHONPATH=ros_ws/src/mns_core:ros_ws/src/mns_navigation:ros_ws/src/mns_motion:ros_ws/src/mns_multi_robot python3 -m pytest -q tests

validate: lint test

build:
	docker compose run --rm robotics-dev bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/hydra_ws/install/setup.bash && colcon build && colcon test && colcon test-result --verbose'

robotics-image:
	docker compose build robotics-dev

model-source:
	bash tools/verify_model_source.sh

robotics-ml-image: robotics-image model-source
	docker compose --profile ml build robotics-ml-dev

models:
	bash tools/fetch_model_assets.sh

test-models: model-source
	docker compose --profile ml run --rm robotics-ml-dev python3 /workspace/tools/smoke_diffusion.py

phase1:
	docker compose --profile simulation --profile robotics up simulation robotics-phase1

phase1-diffusion:
	docker compose --profile simulation --profile ml up simulation robotics-phase1-diffusion

phase2:
	docker compose --profile simulation --profile robotics up simulation robotics-phase2

phase1-synthetic:
	docker compose run --rm robotics-dev ros2 launch mns_bringup phase1.launch.py simulation_mode:=synthetic navigator:=$${MNS_NAVIGATOR:-coverage} run_id:=$${MNS_RUN_ID:-synthetic-phase1}

phase1-synthetic-diffusion:
	docker compose --profile ml run --rm robotics-ml-dev ros2 launch mns_bringup phase1.launch.py simulation_mode:=synthetic navigator:=diffusion run_id:=$${MNS_RUN_ID:-synthetic-diffusion}

phase2-synthetic:
	docker compose run --rm robotics-dev ros2 launch mns_bringup phase2.launch.py simulation_mode:=synthetic navigator:=$${MNS_NAVIGATOR:-configured} run_id:=$${MNS_RUN_ID:-synthetic-phase2}

accept-phase1:
	docker compose run --rm robotics-dev python3 /workspace/tools/runtime_acceptance.py --robots go2_1 --duration $${MNS_ACCEPTANCE_DURATION:-12}

accept-phase2:
	docker compose run --rm robotics-dev python3 /workspace/tools/runtime_acceptance.py --robots go2_1 go2_2 uav_1 uav_2 --duration $${MNS_ACCEPTANCE_DURATION:-12}
