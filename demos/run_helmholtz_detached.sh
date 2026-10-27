#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/run-config.json (activate the Helmholtz environment first)" >&2
    exit 2
fi
config="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1")"
test -f "$config"
command -v setsid >/dev/null
mkdir -p run-logs
logs="$(mktemp -d "$PWD/run-logs/helmholtz-XXXXXX")"
setsid -f bash -c '
    echo $$ > "$2/pid"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    python -u -m dtcc_sim.helmholtz.launch --config "$1" --run --export vtu
    code=$?
    printf "%s\n" "$code" > "$2/exit-code.txt"
    exit "$code"
' helmholtz "$config" "$logs" > "$logs/run.log" 2>&1 < /dev/null
printf 'Detached job launched. Logs and exit code: %s\n' "$logs"
