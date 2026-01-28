#!/bin/bash
set -euo pipefail

# ---------------- Common settings ----------------
TIMEENC=1
MODE=dplr
DEVICE=0
SEED=4444

INITS=(
  safari_morletS
  safari_gaussS
  safari_mexhatS
  safari_dpssS
  safari_morletT
  safari_gaussT
  safari_mexhatT
  safari_dpssT
)

# ---------------- Resume control (Option A: skip-until) ----------------
# Set these to where you want to restart:
START_DATASET="weather"          # etth1 | etth2 | ettm | ecl | weather
START_INIT="safari_dpssT"    # exact init string
FOUND_START=0

# Helper to decide whether to run or skip
should_run () {
  local dataset="$1"
  local init="$2"

  # If already reached start point, run everything
  if [[ "$FOUND_START" -eq 1 ]]; then
    return 0
  fi

  # Not yet reached start point: only start when dataset+init matches
  if [[ "$dataset" == "$START_DATASET" && "$init" == "$START_INIT" ]]; then
    FOUND_START=1
    return 0
  fi

  return 1
}

# ---------------- ETTh (ETTh1 & ETTh2) ----------------
ETTH_EXPERIMENT="forecasting/s4-informer-etth"
ETTH_SIZES=(
  "720 168 24"
  "720 168 48"
  "720 336 168"
  "720 336 336"
  "720 336 720"
)

for VARIANT in 0 1; do
  DATASET_NAME="etth$((VARIANT+1))"

  for INIT in "${INITS[@]}"; do
    for SIZE in "${ETTH_SIZES[@]}"; do
      read SEQ LABEL PRED <<< "$SIZE"

      if ! should_run "$DATASET_NAME" "$INIT"; then
        continue
      fi

      WANDB_NAME="informer/${DATASET_NAME}_safari-${INIT#safari_}_pred${PRED}_seed${SEED}"

      python train.py \
        experiment="$ETTH_EXPERIMENT" \
        dataset.variant="$VARIANT" \
        dataset.size=[$SEQ,$LABEL,$PRED] \
        dataset.timeenc="$TIMEENC" \
        dataset.features="S" \
        model.layer.mode="$MODE" \
        model.layer.init="$INIT" \
        trainer.devices=[$DEVICE] \
        train.seed="$SEED" \
        +wandb.name="$WANDB_NAME"
    done
  done
done

# ---------------- ETTm ----------------
ETTM_EXPERIMENT="forecasting/s4-informer-ettm"
ETTM_SIZES=(
  "96 48 24"
  "96 48 48"
  "384 384 96"
  "384 384 288"
  "384 384 672"
)

for INIT in "${INITS[@]}"; do
  for SIZE in "${ETTM_SIZES[@]}"; do
    read SEQ LABEL PRED <<< "$SIZE"

    if ! should_run "ettm" "$INIT"; then
      continue
    fi

    WANDB_NAME="informer/ettm_safari-${INIT#safari_}_pred${PRED}_seed${SEED}"

    python train.py \
      experiment="$ETTM_EXPERIMENT" \
      dataset.size=[$SEQ,$LABEL,$PRED] \
      dataset.timeenc="$TIMEENC" \
      dataset.features="S" \
      model.layer.mode="$MODE" \
      model.layer.init="$INIT" \
      trainer.devices=[$DEVICE] \
      train.seed="$SEED" \
      +wandb.name="$WANDB_NAME"
  done
done

# ---------------- ECL ----------------
ECL_EXPERIMENT="forecasting/s4-informer-ecl"
ECL_SIZES=(
  # "720 168 24"
  # "720 168 48"
  # "720 336 168"
  # "720 336 336"
  # "720 336 720"
  "960 336 960"
)

for INIT in "${INITS[@]}"; do
  for SIZE in "${ECL_SIZES[@]}"; do
    read SEQ LABEL PRED <<< "$SIZE"

    if ! should_run "ecl" "$INIT"; then
      continue
    fi

    WANDB_NAME="informer/ecl_safari-${INIT#safari_}_pred${PRED}_seed${SEED}"

    python train.py \
      experiment="$ECL_EXPERIMENT" \
      dataset.size=[$SEQ,$LABEL,$PRED] \
      dataset.timeenc="$TIMEENC" \
      dataset.features="S" \
      model.layer.mode="$MODE" \
      model.layer.init="$INIT" \
      trainer.devices=[$DEVICE] \
      train.seed="$SEED" \
      +wandb.name="$WANDB_NAME"
  done
done

# ---------------- Weather ----------------
WEATHER_EXPERIMENT="forecasting/s4-informer-weather"
WEATHER_SIZES=(
  # "720 168 24"
  # "720 168 48"
  # "168 168 168"
  # "336 336 336"
  "720 336 720"
)

for INIT in "${INITS[@]}"; do
  for SIZE in "${WEATHER_SIZES[@]}"; do
    read SEQ LABEL PRED <<< "$SIZE"

    if ! should_run "weather" "$INIT"; then
      continue
    fi

    WANDB_NAME="informer/weather-multivariate_safari-${INIT#safari_}_pred${PRED}_seed${SEED}"

    python train.py \
      experiment="$WEATHER_EXPERIMENT" \
      dataset.size=[$SEQ,$LABEL,$PRED] \
      dataset.timeenc="$TIMEENC" \
      dataset.features="M" \
      model.layer.mode="$MODE" \
      model.layer.init="$INIT" \
      trainer.devices=[$DEVICE] \
      train.seed="$SEED" \
      +wandb.name="$WANDB_NAME"
  done
done
