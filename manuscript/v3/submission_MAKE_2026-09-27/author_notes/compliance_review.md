# MAKE submission preparation review

Prepared 27 September 2026 using all 44 pages of the author-supplied
[Instructions for Authors PDF](</home/ozgur/Desktop/Instructions for Authors _ MAKE _ MDPI.pdf>).
The PDF was saved that same day. Attempts to retrieve the live instructions page
returned HTTP 429, so the supplied PDF is the authoritative requirement snapshot
for this preparation. [Official instructions](https://www.mdpi.com/journal/make/instructions).

The author selected free-format initial submission and supplied the affiliation:
Department of Artificial Intelligence, Adana Alparslan Turkes Science and
Technology University, Adana 01250, Turkey. The author confirmed that the manuscript
is not submitted elsewhere and disclosed an older arXiv preprint.

| Requirement in supplied PDF | Submission treatment |
|---|---|
| Article and journal fit, pp. 1–3 | Article: a learning algorithm, representations, theory and empirical evaluation. The letter explains the fit without claiming an accuracy advantage over classical comparators. |
| Free format, p. 4 | Current article layout and consistent author–year references retained. MDPI numbered references are a revision-stage formatting task. |
| Complete LaTeX ZIP and total uploads under 120 MB, pp. 3–4 | All transitively included TeX, bibliography and figure assets are in `ArrowFlow_sources_MAKE.zip`; a clean extracted build is checked. See `verification.json` for sizes and results. |
| Cover letter and two declarations, pp. 4–5 | New one-page letter uses the current title/findings and discloses the arXiv version and its different title. Exclusivity and author-approval wording are drafted for the author's final review. |
| Complete author affiliation/correspondence, pp. 5, 9 | Both PDFs use the affiliation/postcode supplied in this conversation. Main manuscript explicitly designates correspondence at ozguryilmaz@atu.edu.tr. |
| Abstract approximately 200 words maximum, pp. 9–10 | Original 240-word abstract restored verbatim at the author's explicit request. This exceeds the instructions' approximately 200-word guidance; the original wording is retained by author choice. |
| Three to ten keywords, p. 10 | Eight keywords retained. |
| Research sections, pp. 4, 6, 10–11 | Introduction, method, theory, results, discussion and conclusion present. Section 3 is explicitly titled “Materials and Methods: The Ranking Layer”; Section 7 is “Results”. Section numbers and scientific organization are retained. |
| Funding, contributions, conflicts, data availability, pp. 11–13 | Existing declarations retained; the code availability paragraph now accurately distinguishes the public tag from the supplied later source snapshot. |
| GenAI disclosure in methods and acknowledgments, pp. 10, 12 | Both were already present in v3, with tool/model names, uses and author responsibility. Their factual wording is retained. |
| Supplement inventory, p. 11 | Six sections, 75 tables and five figures identified. Supplement contains contents, table-title and figure-title inventories. Code archive is named separately in the main manuscript. |
| Supplement references also in main manuscript, pp. 13, 20–21 | Added the three previously supplement-only references in relevant passages: OpenML benchmark suites, the Borda mean-proximity interpretation, and the distance-based ranking model. No bibliography records were invented or altered. |
| Figures ZIP, RGB and preferably at least 600 dpi, p. 15 | All eight main and five supplementary figures exported as combined 600-dpi RGB PNGs. Matching vector PDFs and a source/number/checksum manifest are also included. |
| Tables at least 8 pt, p. 15 | Fourteen wide tables use sideways landscape floats. Table S68 uses a two-page landscape longtable. A grid-table spacing issue is corrected. Numerical cells and captions are preserved. |
| Code and reproducibility, pp. 16–21 | Added a local code supplement because the public `v2.0.0-make` tag predates the encoder studies. It contains 274 original source files, plus documentation/manifest, including the experimental modules, protocols and renderers. |
| Reviewer suggestions, p. 31 | Three professional candidates with primary-source contact details are listed in `reviewer_candidates.md`; author checks of relationships remain necessary. |
| Preprints allowed, p. 32 | The [official arXiv record](https://arxiv.org/abs/2604.04087v2) confirms *ArrowFlow: Hierarchical Machine Learning in the Space of Permutations*, v2, 1 June 2026. The current title and expanded work are distinguished in the letter. |
| Single-anonymized review, p. 37 | Author identity and affiliation remain in both PDFs. |

## Review before pressing Submit

1. Review the final manuscript, supplementary material and cover letter. The
   declarations are author statements; the packaging checks do not certify them.
2. The response “not submitted anywhere” establishes current exclusivity. If any
   version was previously submitted to an MDPI journal, add that journal and
   manuscript ID to the cover letter and submission form (pp. 4–5).
3. Confirm that the retained ethics/consent statements accurately describe your
   institution's treatment of the secondary public clinical datasets. The supplied
   instructions distinguish access to data from the basis for exemption (pp. 21–23).
   No new exemption or approval has been invented here.
4. Confirm reviewer independence before selecting the three names; complete any
   contact fields the live form requires. ORCID and biography can be supplied if
   desired; no identifier has been guessed.
5. Confirm the plan for sharing the recorded run directories upon request. The
   code supplement includes source, not the raw datasets, prepared splits or
   predictions. Its README explains these dependencies and the machine-specific
   `RUNS_ROOT`. Historical frozen-commit identity could not be checked from the Git
   objects available in this workspace. A deposited versioned source/data release
   would improve persistence, but none was published during this preparation.

## Scope of verification

This is submission preparation, not a fresh scientific peer review. The existing
experimental results were not rerun. The checks cover compilation, citation and
reference resolution, table numbering and preservation of numerical rows, figure
exports, archive integrity, basic Python syntax, PDF readability and source
portability. The original v3 sources and the PDFs at the parent v3 directory are
preserved; the submission copies are in this dated folder.
