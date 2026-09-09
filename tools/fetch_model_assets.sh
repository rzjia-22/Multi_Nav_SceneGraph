#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${project_root}/models"
source_root="${asset_root}/ForestNavigation"
revision="0b29c399754f510499bfe9cc9d592cba215a7161"

mkdir -p "${asset_root}"
if [[ ! -d "${source_root}/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
    https://github.com/LxRoboticsLab/ForestNavigation.git "${source_root}"
fi

actual_remote="$(git -C "${source_root}" remote get-url origin)"
if [[ "${actual_remote}" != "https://github.com/LxRoboticsLab/ForestNavigation.git" ]]; then
  echo "unexpected ForestNavigation remote: ${actual_remote}" >&2
  exit 1
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${source_root}" checkout "${revision}"
git -C "${source_root}" lfs pull \
  --include="forest_nav/results/epoch=252-step=12903.ckpt,logs/rsl_rl/unitree_go2_rough/2025-11-03_01-41-16/model_5998.pt" \
  --exclude=""

nav_source="${source_root}/forest_nav/results/epoch=252-step=12903.ckpt"
rl_source="${source_root}/logs/rsl_rl/unitree_go2_rough/2025-11-03_01-41-16/model_5998.pt"
echo "023d7b637a3ba2bea6ad918bb45add055ccd57581d09ccf4ed1f1ce56280dfde  ${nav_source}" | sha256sum -c -
echo "f06f93d6606c4cb64a03e2a137db398d52c75b8467268a28cf97bac4faa7086f  ${rl_source}" | sha256sum -c -
ln -sfn "ForestNavigation/forest_nav/results/epoch=252-step=12903.ckpt" "${asset_root}/navdiffusion.ckpt"
ln -sfn "ForestNavigation/logs/rsl_rl/unitree_go2_rough/2025-11-03_01-41-16/model_5998.pt" "${asset_root}/go2_locomotion.pt"

echo "model assets ready in ${asset_root}"

