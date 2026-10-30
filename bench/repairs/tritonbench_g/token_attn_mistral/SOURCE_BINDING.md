# Source binding

Original source SHA256: `39c37c3f43353511cce0d0a3b63fc5e5f1602a41450a51ef4a89065869f8407a`. The original Python file is unchanged.

The repaired JIT retains the historical probability-offset correction and the
original floating product/reduction/accumulator/cast order. Dense checked strides
make the probability element stride one. The wrapper adds metadata and storage
guards, direct output mutation, empty return and the owning CUDA device context.
There are no floating payload copies, private outputs or copyback operations.

For each batch, `first=max(sequence-window,0)`. The independent Real target is the weighted sum of `P[h,attentionStart+n] * V[table[request,first+n],h/(Hq/Hkv),d]` over `n<attentionLength` with `first+n<sequence`. P and V have distinct load masks. In the FP graph a V lane beyond the P tail can still be loaded, so its `0*NaN` product is retained. Attention length need not equal the window length. `B_Start_Loc` is unused; its int32 contents are unrestricted.

The source mapping covers integer metadata, grouped head division, page gathers,
all padded 128-lane products, every loop iteration, native FP32 output cast,
physical stores and their complement. Host-read metadata is the same supplied
metadata launched on the device, under the ordinary no-concurrent-mutation
contract. The checked input description does not assert a desired execution or
output. The independent target uses only initial inputs. Real and FP public
roots are separate; numerical ports assert rounded instruction conformance,
never exact Real arithmetic.

Current evidence: task-local `check_contract.py` and `check_lowering.py`;
`/tmp/veritile-token-mistral-final.log` for final public
Lean builds, and `/tmp/veritile-token-attention-lowering/report.json` for the
eight current-source compiler cases.

The checked public `checked_call` / `checked_fp_call` endpoints compose empty and launch validation. Empty dispatch preserves the supplied descriptor and current memory without canonical inactive strides. Live calls reject mixed FP32/int32 input-byte overlap while retaining same-dtype input aliases. The regression checker uses a fixed random seed; its comparison against a double-precision reference is an approximation check, separate from the word-level FP theorem.
