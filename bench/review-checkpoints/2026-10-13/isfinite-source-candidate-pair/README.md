# Exact complete isfinite source/candidate pair

The source audit validates the complete original fp32/fp64/bf16 quotations
and the complete fp32/fp64 candidate quotations against the pinned Python
helper and writer. The only candidate change is the checked grid tile-index
product, from `j * num_ctas` to `num_ctas * j`. All eleven parameters, both
branches, helper dispatch, casts, rank-one order proofs and boundary checks
remain present. All fifteen candidate mutations are rejected.

The receipt includes exact input bytes and hashes. The original native
matrix still contains 48 rejections; the candidate has not been compiled
natively. This source comparison does not establish native execution or
independent proof acceptance.
