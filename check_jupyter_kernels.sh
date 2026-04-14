#!/usr/bin/env bash
set -euo pipefail

show_env=false

usage() {
  cat <<'EOF'
Usage: ./check_jupyter_kernels.sh [--env]

Affiche les kernels Jupyter en cours d'execution, avec leur heure de demarrage.

Options:
  --env   Affiche aussi les variables d'environnement du process kernel
  -h, --help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env)
      show_env=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Option inconnue: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

runtime_dir="${JUPYTER_RUNTIME_DIR:-}"

if [ -z "$runtime_dir" ] && command -v jupyter >/dev/null 2>&1; then
  runtime_dir="$(jupyter --runtime-dir 2>/dev/null || true)"
fi

if [ -z "$runtime_dir" ]; then
  runtime_dir="${XDG_DATA_HOME:-$HOME/.local/share}/jupyter/runtime"
fi

declare -A kernel_commands=()
declare -A kernel_runtime_files=()

if [ -d "$runtime_dir" ]; then
  shopt -s nullglob
  for kernel_file in "$runtime_dir"/kernel-*.json; do
    pid="$(
      sed -nE 's/^[[:space:]]*"pid"[[:space:]]*:[[:space:]]*([0-9]+),?/\1/p' "$kernel_file" \
        | head -n 1
    )"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
      kernel_commands["$pid"]="${cmd:-kernel vivant}"
      kernel_runtime_files["$pid"]="$(basename "$kernel_file")"
    fi
  done
  shopt -u nullglob
fi

while read -r pid cmd; do
  if [ -n "${pid:-}" ]; then
    kernel_commands["$pid"]="$cmd"
    : "${kernel_runtime_files[$pid]:-}"
  fi
done < <(
  ps -eo pid=,command= \
    | grep -E 'ipykernel_launcher|jupyter-kernel|python[^ ]* .*ipykernel' \
    | grep -v grep \
    || true
)

if [ "${#kernel_commands[@]}" -eq 0 ]; then
  echo "Aucun kernel Jupyter en cours d'execution."
  exit 1
fi

echo "Kernel(s) Jupyter detecte(s) : ${#kernel_commands[@]}"

for pid in $(printf '%s\n' "${!kernel_commands[@]}" | sort -n); do
  start_time="$(ps -p "$pid" -o lstart= 2>/dev/null | sed 's/^[[:space:]]*//')"
  elapsed_seconds="$(ps -p "$pid" -o etimes= 2>/dev/null | tr -d '[:space:]')"
  runtime_file="${kernel_runtime_files[$pid]:-}"

  if [ -z "$runtime_file" ]; then
    runtime_file="$(
      printf '%s\n' "${kernel_commands[$pid]}" \
        | sed -nE 's/.*--f=([^[:space:]]*kernel-[^[:space:]]+\.json).*/\1/p' \
        | xargs -r basename
    )"
  fi

  runtime_file="${runtime_file:-"(runtime introuvable)"}"

  printf ' - PID %s\n' "$pid"
  printf '   Demarre: %s\n' "${start_time:-inconnu}"
  printf '   Age (s): %s\n' "${elapsed_seconds:-inconnu}"
  printf '   Runtime: %s\n' "$runtime_file"
  printf '   Commande: %s\n' "${kernel_commands[$pid]}"

  if [ "$show_env" = true ]; then
    if [ -r "/proc/$pid/environ" ]; then
      echo "   Variables d'environnement:"
      while IFS= read -r env_line; do
        printf '     %s\n' "$env_line"
      done < <(tr '\0' '\n' <"/proc/$pid/environ" | sort)
    else
      echo "   Variables d'environnement: inaccessibles"
    fi
  fi
done

cat <<'EOF'

Note: les variables Python internes du notebook ne sont pas accessibles depuis un simple script bash.
Pour les lister, il faut se connecter au kernel Jupyter via son fichier de runtime et executer du code Python dans ce kernel.
Pour interrompre un notebook sans perdre la memoire chargee, utilisez ./restart_jupyter_kernel.py --list puis ./restart_jupyter_kernel.py --interrupt --pid <PID>.
Pour redemarrer un kernel gere par Jupyter Server, utilisez ./restart_jupyter_kernel.py --list puis ./restart_jupyter_kernel.py ...
EOF
