#!/usr/bin/env bash
# One Kaggle training round from the terminal (run via `make kaggle-run`):
#   1. git push            (so the notebook clones the current code)
#   2. dataset version     (only if the packed .npz files changed since the last upload)
#   3. kernel push         (uploads notebooks/train_kaggle.ipynb and starts a GPU run)
#   4. poll until the run finishes
#   5. download the output to build/models_<timestamp>/ and print the metrics table
# Steps 1-3 ask first (pushing code, uploading data, spending GPU quota). KAGGLE_YES=1 skips
# the prompts; use it only after you have said yes. Needs `kaggle auth login` done once.
set -euo pipefail
cd "$(dirname "$0")/.."

KAGGLE_USER="bhavishya12345678"
DATASET="$KAGGLE_USER/impactlens-packed"
KERNEL="$KAGGLE_USER/impactlens-train"
DATA_DIR="build/kaggle"
STAMP="build/.kaggle_dataset.sha256"  # hash of the last uploaded .npz files
POLL_SECONDS="${POLL_SECONDS:-60}"
MESSAGE="${MESSAGE:-update $(date +%Y-%m-%d)}"
export PYTHONWARNINGS=ignore

confirm() {
  if [[ "${KAGGLE_YES:-0}" == "1" ]]; then
    echo ">> $1: yes (KAGGLE_YES=1)"
    return 0
  fi
  read -r -p ">> $1 [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "stopped."; exit 1; }
}

# Safety: the kernel must stay private.
grep -q '"is_private": true' notebooks/kernel-metadata.json \
  || { echo "notebooks/kernel-metadata.json must have \"is_private\": true"; exit 1; }

# ---- 1. code -----------------------------------------------------------------------------
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Uncommitted changes: commit first, the notebook clones what is on GitHub."; exit 1
fi
git fetch -q origin
ahead=$(git rev-list --count '@{u}..HEAD')
if [[ "$ahead" -gt 0 ]]; then
  confirm "git push $ahead commit(s) to $(git remote get-url origin)"
  git push
else
  echo "== code: GitHub is up to date"
fi

# ---- 2. data (only if changed) -----------------------------------------------------------
grep -q "\"$DATASET\"" "$DATA_DIR/dataset-metadata.json" \
  || { echo "$DATA_DIR/dataset-metadata.json must have id $DATASET"; exit 1; }
hash=$(shasum -a 256 "$DATA_DIR"/*.npz | shasum -a 256 | cut -d' ' -f1)
if [[ -f "$STAMP" && "$(cat "$STAMP")" == "$hash" ]]; then
  echo "== data: unchanged since the last upload, skipping"
else
  size=$(du -ch "$DATA_DIR"/*.npz | tail -1 | cut -f1)
  confirm "upload a new version of $DATASET ($size, stays private)"
  kaggle datasets version -p "$DATA_DIR" -m "$MESSAGE"
  echo "waiting for the new dataset version to be ready ..."
  until kaggle datasets status "$DATASET" 2>/dev/null | grep -q ready; do sleep 20; done
  echo "$hash" > "$STAMP"
  echo "== data: new version ready"
fi

# ---- 3. run ------------------------------------------------------------------------------
confirm "push $KERNEL and start a GPU run (uses GPU quota)"
kaggle kernels push -p notebooks

# ---- 4. wait -----------------------------------------------------------------------------
sleep 30
while true; do
  status=$(kaggle kernels status "$KERNEL" 2>/dev/null | tail -1)
  echo "$(date +%H:%M:%S) $status"
  case "$status" in
    *COMPLETE*|*complete*) break ;;
    *ERROR*|*error*|*CANCEL*|*cancel*) failed=1; break ;;
  esac
  sleep "$POLL_SECONDS"
done

# ---- 5. results --------------------------------------------------------------------------
out="build/models_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
kaggle kernels output "$KERNEL" -p "$out" >/dev/null
echo "== output downloaded to $out"
if [[ "${failed:-0}" == "1" ]]; then
  echo "RUN FAILED. Last lines of the log:"
  log=$(ls "$out"/*.log 2>/dev/null | head -1 || true)
  [[ -n "$log" ]] && uv run python scripts/kaggle_log.py "$log" | tail -40
  exit 1
fi
uv run python scripts/metrics_summary.py "$out"
