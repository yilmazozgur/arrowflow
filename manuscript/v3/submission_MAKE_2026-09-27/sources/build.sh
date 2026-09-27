#!/usr/bin/env bash
# Build the ArrowFlow v3 manuscript (main.tex) and its supplement (supplement.tex).
#
#   ./build.sh [OUTPUT_DIR]      default OUTPUT_DIR: ./build (next to this script)
#
# Each document runs pdflatex, bibtex and pdflatex twice, then reruns pdflatex while the log still asks for a
# rerun, with at most five pdflatex passes per document. The supplement resolves its cross-references through
# xr-hyper (\externaldocument{main}), so main is built first and the two documents share one output directory.
# Sources are read from this directory and only OUTPUT_DIR is written to, so concurrent builds can each use
# their own OUTPUT_DIR. The build fails (exit 1) if a supplement table or subsection number quoted in the main text
# differs from the number LaTeX assigned in the built supplement, if a supplement subsection number typed in the main
# text ("Section~S3.2") is not declared by such a quote, if a supplement section number typed in the sources
# ("Section~S3") is not the number of its pinned supplement section, or if LaTeX reports a float too large for its page
# in either document.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$SRC/build}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$SRC"

MAX_PASSES=5
RERUN='Rerun to get|may have changed\. Rerun|Please rerun LaTeX'

latex_pass() {
  # pdflatex output is discarded; on failure name the log and quote its first error.
  if ! pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" "$1.tex" >/dev/null; then
    echo "pdflatex failed for $1; see $OUT/$1.log" >&2
    grep -n -m 1 -A 3 '^!' "$OUT/$1.log" >&2 || true
    exit 1
  fi
}

bibtex_pass() {
  # BibTeX writes next to the .aux, so run it inside OUT and point it back at the
  # sources for .bib/.bst files. A document without any \citation makes BibTeX
  # exit 2 ("I found no \citation commands"), so skip it in that case; otherwise
  # exit status 1 means warnings only and anything higher is a real error.
  if ! grep -q '^\\citation' "$OUT/$1.aux"; then
    echo "   (no citations in $1; bibtex skipped)"
    rm -f "$OUT/$1.bbl"
    return 0
  fi
  local rc=0
  ( cd "$OUT" && BIBINPUTS="$SRC:" BSTINPUTS="$SRC:" bibtex "$1" >/dev/null ) || rc=$?
  if [ "$rc" -ge 2 ]; then
    echo "bibtex failed for $1 (exit $rc); see $OUT/$1.blg" >&2
    exit "$rc"
  fi
}

build_one() {
  local job="$1" passes=3
  echo "== $job"
  latex_pass "$job"
  bibtex_pass "$job"
  latex_pass "$job"
  latex_pass "$job"
  while grep -q -E "$RERUN" "$OUT/$job.log" && [ "$passes" -lt "$MAX_PASSES" ]; do
    latex_pass "$job"
    passes=$((passes + 1))
  done
  grep -h 'Output written' "$OUT/$job.log" | tail -1
  printf '   pdflatex passes: %s   rerun warnings left: %s\n' "$passes" \
    "$(grep -c -E "$RERUN" "$OUT/$job.log" || true)"
  printf '   undefined references: %s   undefined citations: %s   bibtex warnings: %s\n' \
    "$(grep -c 'LaTeX Warning: Reference' "$OUT/$job.log" || true)" \
    "$(grep -c 'LaTeX Warning: Citation' "$OUT/$job.log" || true)" \
    "$( if [ -f "$OUT/$job.blg" ]; then grep -c '^Warning--' "$OUT/$job.blg" || true; else echo 0; fi )"
}

check_supplement_numbers() {
  # The main text cannot \ref a supplement label, so a quoted supplement table or subsection number is recorded as
  # '% supplement-ref label=number' in the file that quotes it (render_tables.py writes these lines for captions; the
  # section files carry their own). Compare every quote with the built supplement and fail on any mismatch.
  local total=0 bad=0 pair label num
  for pair in $(grep -h '^% supplement-ref' "$SRC"/main.tex "$SRC"/sections/*.tex "$SRC"/tables/*.tex "$SRC"/figures/*.tex 2>/dev/null \
      | sed 's/^% supplement-ref//'); do
    label="${pair%%=*}"
    num="${pair#*=}"
    total=$((total + 1))
    if ! grep -q -F "\\newlabel{$label}{{$num}" "$OUT/supplement.aux"; then
      echo "   quoted supplement number $label=$num does not match $OUT/supplement.aux" >&2
      bad=$((bad + 1))
    fi
  done
  printf '   supplement numbers quoted in the main text: %s checked, %s mismatched\n' "$total" "$bad"
  if [ "$bad" -gt 0 ]; then
    echo "build failed: $bad quoted supplement table number(s) differ from the built supplement;" \
      "re-run render_tables.py and rebuild" >&2
    exit 1
  fi
}

check_typed_subsections() {
  # A supplement subsection number typed in the main text ("Section~S3.2") must be declared by a '% supplement-ref'
  # line of the main-text sources, which check_supplement_numbers compares with the built supplement, so a renumbered
  # supplement subsection fails the build until every typed pointer is rechecked.
  local typed declared num bad=0
  typed="$(cat "$SRC"/main.tex "$SRC"/sections/*.tex "$SRC"/tables/*.tex "$SRC"/figures/*.tex | grep -v '^%' \
    | grep -o -E 'S[0-9]+\.[0-9]+' | sort -u | tr '\n' ' ' || true)"
  declared="$(grep -h '^% supplement-ref' "$SRC"/main.tex "$SRC"/sections/*.tex "$SRC"/tables/*.tex "$SRC"/figures/*.tex \
    2>/dev/null | sed 's/^.*=//' | sort -u | tr '\n' ' ' || true)"
  for num in $typed; do
    case " $declared " in
      *" $num "*) ;;
      *) echo "   Section~$num is typed in the main text but no '% supplement-ref' line declares it" >&2
         bad=$((bad + 1)) ;;
    esac
  done
  printf '   supplement subsections typed in the main text: %s; %s undeclared\n' "${typed% }" "$bad"
  if [ "$bad" -gt 0 ]; then
    echo "build failed: $bad typed supplement subsection number(s) have no '% supplement-ref' declaration;" \
      "declare each and rebuild" >&2
    exit 1
  fi
}

check_supplement_sections() {
  # Main-text prose and captions name supplement sections by typed number ("Section~S3"), which LaTeX cannot check
  # across documents. The top-level supplement sections are pinned to their numbers here, and every typed number must
  # be one of them, so a reordered, inserted or removed supplement section fails until the typed pointers are rechecked.
  local expected='supp:proofs=S1 supp:protocol=S2 supp:results=S3 supp:ablations=S4 supp:reproducibility=S5 supp:learned=S6'
  local bad=0 pair label num quoted
  for pair in $expected; do
    label="${pair%%=*}"
    num="${pair#*=}"
    if ! grep -q -F "\\newlabel{$label}{{$num}" "$OUT/supplement.aux"; then
      echo "   supplement section $label is not numbered $num in $OUT/supplement.aux" >&2
      bad=$((bad + 1))
    fi
  done
  quoted="$(cat "$SRC"/main.tex "$SRC"/sections/*.tex "$SRC"/tables/*.tex "$SRC"/figures/*.tex \
    | grep -o -E 'Sections?~S[0-9]+((,~|,? and~)S[0-9]+)*' | grep -o -E 'S[0-9]+' | sort -u | tr '\n' ' ' || true)"
  for num in $quoted; do
    case " $expected " in
      *"=$num "*) ;;
      *) echo "   Section~$num is typed in the sources but is not a pinned supplement section" >&2
         bad=$((bad + 1)) ;;
    esac
  done
  printf '   supplement sections typed in the sources: %s; %s mismatched\n' "${quoted% }" "$bad"
  if [ "$bad" -gt 0 ]; then
    echo "build failed: typed supplement section numbers do not match the built supplement;" \
      "recheck every Section~S pointer and the pinned list in build.sh" >&2
    exit 1
  fi
}

check_float_sizes() {
  # A float taller than the text height runs into the page footer, where the page number can print over a table cell, and
  # LaTeX only warns about it. Any "Float too large for page" warning in either document fails the build.
  local job n total=0
  for job in main supplement; do
    n="$(grep -c 'Float too large for page' "$OUT/$job.log" || true)"
    total=$((total + n))
    printf '   floats too large for the page in %s: %s\n' "$job" "$n"
    if [ "$n" -gt 0 ]; then
      grep 'Float too large for page' "$OUT/$job.log" | sed "s|^|   $job.log: |" >&2
    fi
  done
  if [ "$total" -gt 0 ]; then
    echo "build failed: $total float(s) too large for the page; shorten or split each float named above and rebuild" >&2
    exit 1
  fi
}

build_one main
build_one supplement
check_supplement_numbers
check_typed_subsections
check_supplement_sections
check_float_sizes
echo "Built $OUT/main.pdf and $OUT/supplement.pdf"

# Keep submission filenames beside the conventional build outputs so remote PDF links work.
cp "$OUT/main.pdf" "$OUT/manuscript_ArrowFlow_MAKE.pdf"
cp "$OUT/supplement.pdf" "$OUT/supplementary_ArrowFlow_MAKE.pdf"
