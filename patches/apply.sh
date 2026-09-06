#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  PATCH_CONTAINER_ARGS=()
  PATCH_SERVER_ARGS=()
  _vllm_patch_output=$(sudo bash "${BASH_SOURCE[0]}" "$@") || return 1
  # shellcheck disable=SC2034
  mapfile -d '' -t PATCH_CONTAINER_ARGS < "$_vllm_patch_output/podman.args" || return 1
  # shellcheck disable=SC2034
  mapfile -d '' -t PATCH_SERVER_ARGS < "$_vllm_patch_output/vllm.args" || return 1
  unset _vllm_patch_output
  return 0
fi

set -euo pipefail
umask 022
if [[ $# != 1 ]]; then
  echo "Usage: $0 IMAGE" >&2
  exit 1
fi

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
site_packages=/usr/local/lib/python3.12/dist-packages
patches=(
  "$here/10-qwen-quantized-embeddings.patch"
  "$here/20-qwen-mtp-quantized-embeddings.patch"
  "$here/30-flashinfer-shared-workspace.patch"
  "$here/40-triton-kv-scale-lifetime.patch"
  "$here/50-flashinfer-int8-xqa.patch"
  "$here/60-qwen-int8-hadamard.patch"
  "$here/70-host-mapped-embeddings.patch"
  "$here/80-gdn-direct-output.patch"
  "$here/90-mrope-cache-cap.patch"
  "$here/100-gdn-state-replay.patch"
)
if ! podman image exists "$1"; then
  podman pull "$1" >&2
fi
image_id=$(podman image inspect --format '{{.Id}}' "$1")
key=$({ printf '%s\n' "$image_id"; cat "${patches[@]}" "$here/apply.sh"; } | sha256sum)
output="$here/.runtime/${key%% *}"
if [[ -f "$output/podman.args" ]]; then
  printf '%s\n' "$output"
  exit 0
fi

mkdir -p "$here/.runtime"
build=$(mktemp -d "$here/.runtime/.build-XXXXXXXX")
trap 'rm -rf -- "$build"' EXIT
awk '/^\+\+\+ b\// { print substr($2, 3) }' "${patches[@]}" | sort -u > "$build/files.list"
awk '/^--- a\// { print substr($2, 3) }' "${patches[@]}" | sort -u > "$build/originals.list"

podman run --rm -i --network=none --security-opt=label=disable \
  -v "$build:/output" --entrypoint python3 "$image_id" - <<'PY' >&2
from importlib.metadata import version
from pathlib import Path
import shutil

assert version("vllm") == "0.26.1rc1.dev1046+gba07e4a48", "Expected pinned vLLM ba07e4a"
assert version("flashinfer-python") == "0.6.17", "Expected FlashInfer 0.6.17"
site = Path("/usr/local/lib/python3.12/dist-packages")
contract = site / "vllm/third_party/flash_linear_attention/ops/chunk_o.py"
assert "o = core_attn_out[: v.numel()].view(*v.shape)" in contract.read_text(), "FLA direct-output contract changed"
output = Path("/output")
for name in (output / "originals.list").read_text().splitlines():
    target = output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(site / name, target)
PY

for patch_file in "${patches[@]}"; do
  patch --directory "$build" --strip=1 --batch --forward --fuzz=0 \
    --no-backup-if-mismatch --reject-file=- < "$patch_file" >&2
done
python3 - "$build" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
for name in (root / "files.list").read_text().splitlines():
    if name.endswith(".py"):
        path = root / name
        compile(path.read_bytes(), str(path), "exec")
PY

replay_dir=vllm/model_executor/layers/mamba/gdn
podman run --rm --network=none --security-opt=label=disable \
  -v "$build:/output" --entrypoint nvcc "$image_id" \
  -std=c++17 -O3 --use_fast_math -arch=sm_120 \
  --shared -Xcompiler=-fPIC \
  "/output/$replay_dir/replay.cu" -o "/output/$replay_dir/libgdn_replay.so" >&2
printf '%s\n' "$replay_dir/libgdn_replay.so" >> "$build/files.list"

{
  while IFS= read -r path; do
    printf '%s\0' -v "$output/$path:$site_packages/$path:ro"
  done < "$build/files.list"
  printf '%s\0' \
    -e "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((16 * 1024 * 1024))" \
    -e VLLM_FLASHINFER_INT8_PTH_NATIVE=1 \
    -e VLLM_GDN_DIRECT_SCAN_OUTPUT=1 \
    -e VLLM_USE_HOST_MAPPED_EMBEDDINGS=1 \
    -e VLLM_MROPE_CACHE_CAP=1
} > "$build/podman.args"
printf '%s\0' --kv-cache-dtype int8_per_token_head > "$build/vllm.args"
chmod -R a+rX "$build"
mv -T -- "$build" "$output"
printf '%s\n' "$output"
