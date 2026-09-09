#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${project_root}/models/ForestNavigation"
revision="0b29c399754f510499bfe9cc9d592cba215a7161"

if [[ ! -d "${source_root}/.git" ]]; then
  echo "ForestNavigation source is missing; run 'make models' first" >&2
  exit 1
fi

actual_revision="$(git -C "${source_root}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${revision}" ]]; then
  echo "ForestNavigation revision mismatch: ${actual_revision}" >&2
  exit 1
fi

if [[ -n "$(git -C "${source_root}" status --porcelain -- forest_nav/navdiffusion forest_nav/setup.py)" ]]; then
  echo "ForestNavigation model source has local modifications" >&2
  exit 1
fi

echo "ForestNavigation model source verified at ${revision}"
