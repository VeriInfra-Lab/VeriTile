# layer_norm_ops

The repaired [Python implementation](../layer_norm_ops.py) is copied into [Kernels.lean](../Kernels.lean) as complete forward and backward function bodies. Pointer storage, runtime scalar bindings and constexpr flags are supplied outside the quotations. The source branches, integer-to-FP conversions, load padding, saved statistics, bounded row loop, residual streams, recompute path and stores are retained.

[Executable candidates](../layer_norm_ops.py) supply Python sources for the two existing Lean transformations: reuse the forward column mask and reuse `row_start` in the backward loop bound. These candidate sources were introduced from the repaired Python bodies; they are not historical benchmark originals. All four body pairs contain 18 top-level Python statements.

The composed backward invocation supplies the saved Mean pointer under the existing Python input checks. The pasted body keeps its `Mean is not None` expression and zero fallback. Caller-side pointer presence selects the applicable branch after checking guarded pointer uses; an untaken branch does not load memory.

[Correct.lean](../RealCorrect.lean) states independent LN/RMS, affine, residual, recompute and parameter-gradient formulas with named multi-output observations. [FPEquiv.lean](../FPEquiv.lean) compares each complete source/candidate invocation under its public IO and alias conditions. Empty backward loops and exterior frames remain in scope.

[Contract](../Contract.lean) · [IO](../IO.lean) · [Proof helpers](.). [Validation records](../../../../validation/body-migration-progress.json) distinguish local source/proof checks from independent replay acceptance. Compilation or body comparison alone does not mark this task complete.
