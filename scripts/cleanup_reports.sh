#!/usr/bin/env bash
# cleanup_reports.sh (robusto)
# Uso:
#   bash /workspace/scripts/cleanup_reports.sh
#   DEBUG=1 bash /workspace/scripts/cleanup_reports.sh  # modo verboso

set -Eeuo pipefail
trap 'echo "[ERRO] Linha $LINENO: comando falhou"; exit 1' ERR

# Modo debug
if [[ "${DEBUG:-0}" == "1" ]]; then set -x; fi

# Proteções de glob
shopt -s nullglob dotglob

ROOT="/workspace"
RPT="${ROOT}/reports"
TS="$(date +%Y%m%d_%H%M%S)"
ARCH="${ROOT}/_reports_${TS}"
TARGZ="${ARCH}.tar.gz"
SHA="${ARCH}.sha256"

mkdir -p "${ARCH}"

mv_safe() {
  # move args... para destino (último arg); só executa mv se houver fonte existente
  local dest="${@: -1}"
  local sources=("${@:1:$#-1}")
  local present=()
  for s in "${sources[@]}"; do
    [[ -e "$s" ]] && present+=("$s")
  done
  if (( ${#present[@]} )); then
    mv -v "${present[@]}" "$dest"
  fi
}

# 1) mover pastas pesadas (se existirem)
for d in plots figs preds tuned sets literature_fs_baseline; do
  [[ -d "${RPT}/${d}" ]] && mv -v "${RPT}/${d}" "${ARCH}/"
done

# 2) mover bundles/snapshots/figs soltos (sem abortar se não houver)
mv_safe "${RPT}"/*bundle*.tar.gz "${ARCH}/"
mv_safe "${RPT}"/UFMS_snapshot_* "${ARCH}/"
mv_safe "${RPT}"/scatter_*.png "${ARCH}/"

# 3) estado antes da compactação
echo "[FICOU em ${RPT}]"
ls -lah "${RPT}" || true
echo "[FOI para ${ARCH}]"
find "${ARCH}" -maxdepth 2 -type d -print | sort

# 4) se o ARCH estiver vazio, pular tar
if [[ -z "$(find "${ARCH}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Nada para arquivar (archive vazio). Limpando diretório temporário e saindo."
  rmdir "${ARCH}" 2>/dev/null || true
else
  # 4a) compactar + validar
  tar -czf "${TARGZ}" -C "${ROOT}" "$(basename "${ARCH}")"
  du -h "${TARGZ}"
  tar -tzf "${TARGZ}" >/dev/null && echo "OK: tar.gz íntegro"

  # 5) checksum e symlink 'latest'
  sha256sum "${TARGZ}" | tee "${SHA}"
  ln -sfn "$(basename "${TARGZ}")" "${ROOT}/_reports_latest.tar.gz"

  # 6) apagar pasta expandida (mantém só .tar.gz)
  rm -rf "${ARCH}"
fi

# 7) verificador final: só o que queremos no /reports
echo "[Somente o que queremos em ${RPT}]"
find "${RPT}" -maxdepth 1 -mindepth 1 \
  ! -name "finals_cv" ! -name "winners" ! -name "progress" ! -name "RUN_LOG.md" -print || true

echo "DONE."
