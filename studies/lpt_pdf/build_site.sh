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
  printf '<p style="margin:0;padding:10px 16px;font:14px/1.4 sans-serif;background:#f3f2ef;color:#1b1b1a">DiscoDJ.jl studies: this report (LPT vs N-body before shell crossing) · <a href="studies/eft_shell_crossing/">UV sensitivity of long modes across shell crossing</a></p>\n'
  cat "$here/report.html"
  printf '\n</body>\n</html>\n'
} > "$out/index.html"
cp "$here"/figures/*.png "$out/figures/"
cp "$here"/copula/figures/*.png "$out/copula/figures/"
cp "$here"/sheet_pdf/figures/*.png "$out/sheet_pdf/figures/"
# the report used to be served from the source path; keep that address working
mkdir -p "$out/studies/lpt_pdf"
printf '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=../../"><link rel="canonical" href="../../"><a href="../../">The report has moved</a>\n' > "$out/studies/lpt_pdf/report.html"
touch "$out/.nojekyll"
echo "site written to $out"
