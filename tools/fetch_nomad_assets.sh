#!/usr/bin/env bash
set -euo pipefail

# 获取并校验官方 NoMaD 源码与公开 checkpoint。二者都是本地运行资产，
# 保持在 Git 忽略的 models/NoMaD 中，避免把第三方仓库和大权重提交进本仓库。
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${project_root}/models/NoMaD"
repository_url="https://github.com/robodhruv/visualnav-transformer.git"
source_revision="dca79815b704e5aa9c6bdc3082351f9e3b2848c2"
checkpoint_path="${source_root}/nomad.pth"
checkpoint_url="https://drive.usercontent.google.com/download?id=1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb&export=download&confirm=t"
checkpoint_sha256="70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333"
checkpoint_size_bytes="76473631"

if [[ ! -d "${source_root}/.git" ]]; then
  if [[ -e "${source_root}" ]]; then
    echo "Refusing to replace non-Git path: ${source_root}" >&2
    exit 1
  fi
  git clone "${repository_url}" "${source_root}"
fi

if ! git -C "${source_root}" diff --quiet || ! git -C "${source_root}" diff --cached --quiet; then
  echo "Official NoMaD checkout has local tracked changes; refusing to switch revision." >&2
  exit 1
fi

actual_remote="$(git -C "${source_root}" remote get-url origin)"
if [[ "${actual_remote}" != "${repository_url}" ]]; then
  echo "Unexpected NoMaD origin: ${actual_remote}" >&2
  exit 1
fi
if ! git -C "${source_root}" cat-file -e "${source_revision}^{commit}" 2>/dev/null; then
  git -C "${source_root}" fetch --quiet origin "${source_revision}"
fi
git -C "${source_root}" checkout --quiet --detach "${source_revision}"

checkpoint_matches=false
if [[ -f "${checkpoint_path}" ]]; then
  actual_size="$(stat -c '%s' "${checkpoint_path}")"
  actual_sha256="$(sha256sum "${checkpoint_path}" | awk '{print $1}')"
  if [[ "${actual_size}" == "${checkpoint_size_bytes}" && "${actual_sha256}" == "${checkpoint_sha256}" ]]; then
    checkpoint_matches=true
  fi
fi

if [[ "${checkpoint_matches}" != true ]]; then
  temporary_path="$(mktemp "${checkpoint_path}.download.XXXXXX")"
  cleanup() { rm -f -- "${temporary_path}"; }
  trap cleanup EXIT
  curl --fail --location --retry 3 --output "${temporary_path}" "${checkpoint_url}"
  actual_size="$(stat -c '%s' "${temporary_path}")"
  actual_sha256="$(sha256sum "${temporary_path}" | awk '{print $1}')"
  if [[ "${actual_size}" != "${checkpoint_size_bytes}" || "${actual_sha256}" != "${checkpoint_sha256}" ]]; then
    echo "Downloaded NoMaD checkpoint failed size/SHA256 verification." >&2
    exit 1
  fi
  mv -- "${temporary_path}" "${checkpoint_path}"
  trap - EXIT
fi

echo "NoMaD source ${source_revision} and checkpoint ${checkpoint_sha256} are ready."
