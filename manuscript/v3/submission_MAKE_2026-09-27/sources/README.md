# ArrowFlow: MAKE free-format submission sources

Prepared 27 September 2026 from the author's v3 manuscript.

Run `bash build.sh` from this directory with a complete TeX Live installation,
pdfLaTeX and BibTeX. The script builds the main manuscript first, then the
supplement, resolves citations and cross-references, and checks supplementary
numbering and oversized floats. No experiment runs or network access are needed
to build the documents. Figures are included as editable TikZ sources or vector PDFs.

Outputs are in `build/`. Upload `manuscript_ArrowFlow_MAKE.pdf` and
`supplementary_ArrowFlow_MAKE.pdf`. Keep those names together when viewing locally:
the supplement's links to the main manuscript use the first name. Conventional
`main.pdf` and `supplement.pdf` outputs are also retained. Browser PDF viewers may
not follow links to another local PDF; desktop PDF viewers generally support them.

The two `.bbl` files are included for editorial convenience; `build.sh` regenerates
them from `references_v3.bib`. The free-format submission uses consistent
author–year references. MDPI's numbered style can be applied if revision is invited.

The separately submitted `ArrowFlow_code_MAKE.zip` contains scientific software;
it is not needed to compile this LaTeX archive.
