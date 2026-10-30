# masked_select

- Name:

masked_select_stable_correctness

Statement:

Spec.Real (ReaderIO.view rest p config ⊨{ReaderIO.prefixes p} fun xs j => (((List.range p.size).filter xs.selected).map xs.value)[j.val]?.getD (0 : ℝ))

- [masked_select_repaired.py](masked_select_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected configuration grid, every active stable selected output and exact exterior frame. The wrapper-prepared FP32/Bool/Int64 prefix buffers are the theorem entry, not unprepared raw broadcasting inputs.

## Conditions

- config in configs and KernelSeparate p remain public. All twenty existing source configurations are retained.
- IO.Prefixes remains the independent inclusive-count relation on logical prefix inputs. Exact InputCells retains payload/mask/prefix storage identity.
- Output active extent remains count(mask,size); capacity remains size. Contract separates raw input checks from prepared pointer alignment and fresh-output storage.

Exact FP program pairs and numerical dependencies are recorded in the [masked_select contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
