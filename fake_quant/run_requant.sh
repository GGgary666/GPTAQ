#!/usr/bin/env bash
# Paper-faithful ReQuant runs on Qwen3-1.7B (stage 1) or Llama-3 8B (stage 2).
# Defaults match arXiv:2608.07019: T=4, K=2, WikiText-2 512 x 2048,
# per-channel asymmetric weights, QuaRot, Following GPTAQ.
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATASET_DIR="${DATASET_DIR:-}"
NSAMPLES="${NSAMPLES:-512}"
SAVE_ROOT="${SAVE_ROOT:-./runs}"
SWEEPS="${SWEEPS:-4}"
INIT="${INIT:-rtn}"
ABITS="${ABITS:-16}"
FORMAT="${FORMAT:-int}"
SEQLEN="${SEQLEN:-}"
EVAL_NSAMPLES="${EVAL_NSAMPLES:-}"

COMMON=(
  --model "${MODEL}"
  --nsamples "${NSAMPLES}"
  --cal_dataset wikitext2
  --eval_dataset wikitext2
  --rotate
  --w_bits 4
  --w_groupsize -1
  --w_asym
  --w_clip
  --a_asym
  --k_bits 16
  --v_bits 16
  --requant
  --requant_sweeps "${SWEEPS}"
  --requant_neighborhood 2
  --requant_fp_branch true_fp
  --requant_chunk 32
)
if [[ -n "${DATASET_DIR}" ]]; then
  COMMON+=(--dataset_dir "${DATASET_DIR}")
fi

run() {
  local name="$1"
  shift
  local log="${SAVE_ROOT}/${name}.log"
  mkdir -p "${SAVE_ROOT}"
  echo "=== ${name} ==="
  "${PYTHON}" main.py --save_name "${name}" "${COMMON[@]}" "$@" 2>&1 | tee "${log}"
}

# Usage: INIT=rtn|gptq|gptaq  ABITS=16|4  FORMAT=int|nvfp4  [A_PER_TENSOR=1]
extra=()
if [[ "${INIT}" == "rtn" ]]; then
  extra+=(--w_rtn)
elif [[ "${INIT}" == "gptaq" ]]; then
  extra+=(--asym_calibrate --enable_aq_calibration)
elif [[ "${INIT}" == "gptq" ]]; then
  extra+=()
else
  echo "INIT must be rtn|gptq|gptaq" >&2
  exit 1
fi

if [[ "${FORMAT}" == "nvfp4" ]]; then
  extra+=(--w_format nvfp4)
  if [[ "${ABITS}" == "4" ]]; then
    extra+=(--a_bits 4 --a_format nvfp4 --enable_aq_calibration)
  else
    extra+=(--a_bits 16 --a_format int)
  fi
elif [[ "${ABITS}" == "4" ]]; then
  extra+=(--a_bits 4 --a_clip_ratio 0.9 --enable_aq_calibration --a_format int --w_format int)
else
  extra+=(--a_bits 16 --a_format int --w_format int)
fi

if [[ "${A_PER_TENSOR:-0}" == "1" ]]; then
  extra+=(--a_per_tensor)
fi
if [[ -n "${SEQLEN}" ]]; then
  extra+=(--seqlen "${SEQLEN}")
fi
if [[ -n "${EVAL_NSAMPLES}" ]]; then
  extra+=(--eval_nsamples "${EVAL_NSAMPLES}")
fi

run "${INIT}_w4a${ABITS}_${FORMAT}_requant_t${SWEEPS}k2" "${extra[@]}"
