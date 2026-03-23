#!/usr/bin/env bash
# Push deduplicated judged response JSONL files from experiment 4_2 to a Modal Volume.
#
# For each eval (identity_inference, introspection, misalignment), finds all
# responses_*_judged.jsonl files, deduplicates by prefix (keeping only the
# latest timestamp per prefix), then uploads via `modal volume put`.
#
# File naming patterns:
#   identity_inference: responses_{model}_{suffix}_temp{T}_{YYYYMMDD_HHMMSS}_judged.jsonl
#   introspection:      responses_{model}_{suffix}_temp{T}_{YYYYMMDD_HHMMSS}_judged.jsonl
#   misalignment:       responses_{CAT}_{model}_{suffix}_temp{T}_{YYYYMMDD_HHMMSS}_judged.jsonl
#
# Usage:
#   bash push_results_to_modal.sh              # dry run (default)
#   bash push_results_to_modal.sh --upload     # actually upload
#   bash push_results_to_modal.sh --cleanup    # delete outdated local response files
#   bash push_results_to_modal.sh --volume X   # custom volume name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

VOLUME="entangled-persona"
REMOTE_PREFIX="/4_2_hitler_persona/results"
DRY_RUN=true
CLEANUP=false

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload)   DRY_RUN=false; shift ;;
        --cleanup)  CLEANUP=true; shift ;;
        --volume)   VOLUME="$2"; shift 2 ;;
        --prefix)   REMOTE_PREFIX="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Filter & deduplicate ──────────────────────────────────────────────────
# 1. Only keep files matching the current naming convention, which always
#    contains _base_temp or _ft_temp.  This skips old results that used
#    hash strings, "final", checkpoint numbers, etc.
# 2. Deduplicate by prefix (everything before _YYYYMMDD_HHMMSS_judged.jsonl),
#    keeping only the latest timestamp per prefix.

VALID_PATTERN='_(base|ft)_temp'

deduplicate() {
    local dir="$1"
    declare -A latest  # prefix -> filename

    # Process files in sorted order so the last (latest timestamp) wins
    while IFS= read -r file; do
        basename="$(basename "$file")"
        # Skip files that don't follow the current naming convention
        if ! echo "$basename" | grep -qE "$VALID_PATTERN"; then
            continue
        fi
        # Strip _YYYYMMDD_HHMMSS_judged.jsonl to get the prefix
        prefix="$(echo "$basename" | sed -E 's/_[0-9]{8}_[0-9]{6}_judged\.jsonl$//')"
        latest["$prefix"]="$file"
    done < <(find "$dir" -maxdepth 1 -name 'responses_*_judged.jsonl' -type f | sort)

    # Output deduplicated file paths
    for file in "${latest[@]}"; do
        echo "$file"
    done | sort
}

# ── Collect files across all 3 eval subdirs ───────────────────────────────
SUBDIRS=("identity_inference" "introspection" "misalignment")

all_files=()
total_before=0
for subdir in "${SUBDIRS[@]}"; do
    dir="$RESULTS_DIR/$subdir"
    [[ -d "$dir" ]] || continue

    before=$(find "$dir" -maxdepth 1 -name 'responses_*_judged.jsonl' -type f | wc -l)
    total_before=$((total_before + before))

    while IFS= read -r f; do
        all_files+=("$f")
    done < <(deduplicate "$dir")
done

# ── Cleanup: remove outdated local files ──────────────────────────────────
if $CLEANUP; then
    # Build a set of files to keep (the deduplicated ones)
    declare -A keep_set
    for f in "${all_files[@]}"; do
        keep_set["$f"]=1
    done

    # Find ALL response files (judged and non-judged) and mark those not in keep_set
    stale_files=()
    for subdir in "${SUBDIRS[@]}"; do
        dir="$RESULTS_DIR/$subdir"
        [[ -d "$dir" ]] || continue
        while IFS= read -r f; do
            if [[ -z "${keep_set[$f]+x}" ]]; then
                stale_files+=("$f")
            fi
        done < <(find "$dir" -maxdepth 1 -name 'responses_*.jsonl' -type f | sort)
    done

    echo "=== Experiment 4_2 — Cleanup outdated response files ==="
    echo "Keeping ${#all_files[@]} latest files (from $total_before total)"
    echo ""

    if [[ ${#stale_files[@]} -eq 0 ]]; then
        echo "No outdated files to remove."
    else
        echo "Outdated files to remove (${#stale_files[@]}):"
        for f in "${stale_files[@]}"; do
            echo "  ${f#"$RESULTS_DIR"/}"
        done

        echo ""
        read -rp "Delete these ${#stale_files[@]} files? [y/N] " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            for f in "${stale_files[@]}"; do
                rm "$f"
            done
            echo "Deleted ${#stale_files[@]} outdated files."
        else
            echo "Aborted — no files deleted."
        fi
    fi
    exit 0
fi

echo "=== Experiment 4_2 — Push results to Modal Volume ==="
echo "Volume:        $VOLUME"
echo "Remote prefix: $REMOTE_PREFIX"
echo "Total files:   ${#all_files[@]} (deduplicated from $total_before)"
echo ""

# ── List files ────────────────────────────────────────────────────────────
for f in "${all_files[@]}"; do
    rel="${f#"$RESULTS_DIR"/}"
    echo "  $rel"
done

if $DRY_RUN; then
    echo ""
    echo "Dry run — no files uploaded. Pass --upload to upload."
    exit 0
fi

# ── Upload each subdir ────────────────────────────────────────────────────
# modal volume put uploads a local path to a remote path on the volume.
# We upload per-subdir to preserve directory structure.

echo ""
echo "Uploading..."

# Create a temp staging dir with only the deduplicated files, preserving subdir structure
STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

for f in "${all_files[@]}"; do
    rel="${f#"$RESULTS_DIR"/}"
    subdir="$(dirname "$rel")"
    mkdir -p "$STAGING_DIR/$subdir"
    cp "$f" "$STAGING_DIR/$rel"
done

for subdir in "${SUBDIRS[@]}"; do
    staging_subdir="$STAGING_DIR/$subdir"
    [[ -d "$staging_subdir" ]] || continue

    count=$(find "$staging_subdir" -type f | wc -l)
    remote_dest="$REMOTE_PREFIX/$subdir/"
    echo "  $subdir: $count files -> $remote_dest"

    modal volume put -f "$VOLUME" \
        "$staging_subdir/" \
        "$remote_dest"
done

echo ""
echo "Done. Verify with:"
echo "  modal volume ls $VOLUME $REMOTE_PREFIX/"
