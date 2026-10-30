# Current kernel review: admitted grid y overflow

The selected source is the exact `contract_revision.source` from
`validation/task_progress.json`:
`token_attn_mistral_repaired.py`, SHA256
`36afb09dac2c46a8049cf250c8c592484147e3d6133ad915e6917d8920c9c553`.
The Python source, guards and previous Lean proofs are unchanged.

The kernel is a gathered weighted sum, with no softmax. Set
`first = max(sequence - sliding_window, 0)`. Its independent Real result is
the sum of `P[h, attention_start+n] * V[table[request, first+n], h/(Hq/Hkv), d]`
over `n < attention_length` with `first+n < sequence`. The native interpretation
retains all 128-token products, each complete reduction tree, the FP32 loop
additions and final same-format output cast. P and V have independent masks;
an active V NaN beyond the P tail still contributes a `0*NaN` product.

The selected domain is contiguous ordinary FP32 P/V/Out on one CUDA device,
signed-int32 metadata and table, D in {16,32,64,128}, positive Hq divisible by
positive Hkv, checked integer contents and padded address envelopes, aligned
nonempty buffer views, and output/read byte separation. Same-dtype read aliases
remain admitted. `B_Start_Loc` is unused and its contents remain unrestricted.
The scalar primitive and compiler/reduction choices are explicit model
boundaries. This review adds no finiteness guard or numerical experiment.

## Counterexample in the current admitted domain

Use B=1, Hq=65536, Hkv=1, D=16:

- P has shape [65536,1] and every value is FP32 2.
- V has shape [1,1,16] and every value is FP32 2.
- Out has shape [1,65536,16], initially FP32 7 (4 MiB).
- The int32 table is [[0]]; request=[0], sequence=[1],
  attention_start=[0], attention_length=[1], and sliding_window=1.
- The unused start vector is [-2147483648]. All buffers have distinct,
  aligned, contiguous backing storage.

The unchanged `_checked_arguments` returns True. The unchanged wrapper submits
`grid=(1,65536)` using `heads=prob.shape[0]`. CUDA C++ Programming Guide 12.6.3,
Table 21, gives 65535 as the maximum y- or z-dimension of a grid. Thus this
admitted call requests an architecturally invalid launch. The independent
formula would produce 4 in every output coordinate if it executed; no GPU
output or runtime error was observed by this review.

The grid product is only 65536, and all scalar launch strides and checked
signed32 envelopes are small. This is an individual y-axis limit violation,
separate from the deferred launcher grid-product compatibility issue.

## Evidence and scope

`.veritile-cache/token-attn-mistral-kernel-evidence/launch_probe.py` executes the
actual selected guard and wrapper over real CPU Torch storage with the CUDA
admission property and device context adapted. Large-head launches are captured
without running the kernel. Heads 1, 65535 and 65536 are all admitted, with their
exact launch tuples and buffer descriptors retained in `launch-probe-report.json`.

Five small calls execute the complete original JIT through the CPU interpreter:
zero length, one token, two 128-token tiles, P/V tail mismatch, and zero window.
Every output and read-input frame is checked. A sixth control confirms the
distinct-mask `0*NaN` behavior. These controls establish no GPU numerical
conformance. The official grid-limit HTML and extracted table are reused by
hash from the previously frozen NVIDIA documentation evidence.

Full admitted-domain CUDA success is not established. The existing shared DSL,
Contract/Correct/FPEquiv, Real/native models and logical store/frame proofs
remain preserved; their abstract full-grid success does not establish legal
CUDA launch dimensions. No new Lean build or offline compilation is needed to
establish this counterexample. No head-bound repair, alternate launch,
allocator requirement, returned-object work or unified-byteheap gate is added.

The frozen delivery is
`.veritile-cache/token-attn-mistral-kernel-evidence/delivery.json`.
Root owns independent review, classification, any later repair decision and
the progress ledger.
