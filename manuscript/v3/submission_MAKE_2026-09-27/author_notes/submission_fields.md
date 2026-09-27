# Fields for the MAKE submission form

| Field | Text |
|---|---|
| Journal | Machine Learning and Knowledge Extraction (MAKE) |
| Article type | Article |
| Title | ArrowFlow: Training Ranking Filters with Position Votes |
| Sole author / corresponding author | Ozgur Yilmaz |
| Department | Department of Artificial Intelligence |
| Institution | Adana Alparslan Turkes Science and Technology University |
| City / postcode / country | Adana / 01250 / Turkey |
| Email | ozguryilmaz@atu.edu.tr |
| Keywords | ranking filters; permutation learning; Borda count; Arrow's impossibility theorem; Spearman's footrule; nearest-neighbor classification; alternatives to backpropagation; target propagation |
| Funding, retained from manuscript | This research received no external funding. |
| Conflicts, retained from manuscript | The author declares no conflicts of interest. |
| Preprint DOI | https://doi.org/10.48550/arXiv.2604.04087 |
| Preprint version | https://arxiv.org/abs/2604.04087v2 |
| Earlier title | ArrowFlow: Hierarchical Machine Learning in the Space of Permutations |
| Earlier version date | 1 June 2026 |
| Current consideration elsewhere | No, confirmed by the author in this conversation. |
| Previous MDPI submission | Author to answer from actual history; current exclusivity alone does not answer this field. |
| ORCID | Optional; not supplied. |

## Abstract (240 words)

ArrowFlow is a classifier whose layers work with rankings. Each layer holds learned rankings, the ranking filters, and outputs them sorted by footrule distance to its input. Training examples vote on where a filter's items should go. A weighted Borda count merges the votes, with no derivative computed. The position displacements (motions) returned by the output layer's votes train the hidden layers. A fixed encoder sorts projected features, and a nearest-neighbor vote reads the last hidden ranking. Under nested cross-validation on seventeen datasets, seven used during development and ten chosen by prespecified rules, training the ranking layers improves performance. Scrambled, the motions leave the network less accurate on most datasets than never moving its hidden filters. Turned into target scores, motions of the same kind also train a neural encoder network by gradient steps, without differentiating the sort. As a classifier, the trained encoder network is more accurate than the same network untrained on all 17, significantly on 11. Swapped into ArrowFlow, it raises mean accuracy on 11 datasets, significantly on one, and lowers it significantly on none. Training a neural network via discrete motion signals from a ranking layer is novel and one of the most important contributions of the paper. Counting the filter's current order as one more voter, the update rule violates independence of irrelevant alternatives by Arrow's theorem under a mild weight condition. So a filter's order of two items can change when a third moves.

## Preprint explanation

An earlier version appeared as arXiv:2604.04087v2 under a different title. The
present submission provides revised analysis, nested evaluations, matched learning
controls and the learned-encoder extension. The arXiv version has not been updated
as part of this preparation.

## Choices to make in the live form

Choose the appropriate journal section or special issue, if any; the preparation
does not assume a special-issue invitation. Enter proposed/excluded reviewers in
the form. Select open peer review and alternative-journal transfer options only
according to your preferences. Check the journal's current APC, any institutional
discount and billing details before accepting the submission terms:
https://www.mdpi.com/journal/make/apc . No fee amount is assumed here.
