# Original-program scope review: 19 additional tasks

Whole-original-program acceptance remains **0/183**, with **183 pending**.
The ledger now distinguishes **85 tasks with confirmed structural gaps** and
**98 awaiting a full scope review**. This batch adds 19 reviews, checks 58 public
statements and records 42 specific structural gaps. Existing proofs are preserved.

The cohort consists of 17 frozen batch-02 reviews plus root reviews of
`quant_transpose_kernel` and `swiglu_backward`. Every task's full current Lean/Python
input set, exact public declarations and quoted evidence ranges were rebound to
the clean integration base. The 164 other task rows are unchanged. Producer
receipts, notes, input records and validator are retained verbatim under evidence.

The findings distinguish complete kernel results from omitted original stages.
Examples include the final `torch.mm` in dequantize_matmul, host final logsumexp,
the general matrix-vector loop beyond the proved one-block result, independent
input/output dtype routes and complete wrapper/grid composition. Current complete
rmsnorm loops, reverse-scan tail semantics, embedding vocabulary branches, and
source-ordered arithmetic remain credited. Ordinary bounds/layout premises are
not themselves structural gaps. No new numerical deferral was registered.

`current-declaration-index.json` separately records actual locations and file hashes
for all 938 current public declarations. Historical catalogs and the original
metadata baseline were not rewritten. The current catalog was not regenerated.

The scope CLI actually exited 1 with exactly 183 pending scopes and no other
population, catalog or binding error. No Lean, comparator, frontend or GPU run
was repeated for this source-preserving review. Prior build/global-gate results
remain in `../relu-source-grid-and-exact-numerics/`.
