#!/bin/bash
# Build the static site for GitHub Pages from the study report:
#   studies/lpt_pdf/build_site.sh <outdir>
# report.html is written as page content (no <html>/<head>); this wraps it in a complete
# document and copies the figures it references.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:?usage: build_site.sh <outdir>}"
mkdir -p "$out/figures" "$out/copula/figures" "$out/sheet_pdf/figures"
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
  printf '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
  printf '<style>*,*::before,*::after{box-sizing:border-box}body{margin:0}img{max-width:100%%}[hidden]{display:none!important}</style>\n'
  printf '</head>\n<body>\n'
  cat "$here/report.html"
  printf '\n</body>\n</html>\n'
} > "$out/index.html"
cp "$here"/figures/*.png "$out/figures/"
cp "$here"/copula/figures/*.png "$out/copula/figures/"
cp "$here"/sheet_pdf/figures/*.png "$out/sheet_pdf/figures/"
touch "$out/.nojekyll"
echo "site written to $out"
