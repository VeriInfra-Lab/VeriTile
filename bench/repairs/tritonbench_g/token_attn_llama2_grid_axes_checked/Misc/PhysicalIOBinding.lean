import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.LegacyBinding
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.NativeReference

/-! Optional actual Tensor/physical-memory binding to the shared kernel IO.
This module imports neither public proof entry nor TensorBridge. The mains do
not depend on it. All original physical pointer/alias conditions are inherited. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.PhysicalIOBinding
open VeriTile Triton FP.Structural Whole KernelNative
noncomputable section

theorem real_inputs (t : TensorCall) (words : Nat → Word) (before : Cells ℝ)
    (rest : State ℝ) (contract : InputContract t words) (launch : active t)
    (work : List Program) :
    (Interface.io (fun _ => realAlgebra) t.call (metadata t words) (idealScale t.call.dim)
      work (fun _ _ x => x) id).InputBinding
        (LegacyBinding.values t.call (metadata t words) (buffers t before))
        (initial t.call (metadata t words) (buffers t before) rest) :=
  LegacyBinding.input _ _ _ _ work _ _ _ rest (valid t words contract launch)

theorem native_inputs (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (t : TensorCall) (before : Cells Word) (rest : State Word)
    (contract : InputContract t (wordRead before)) (launch : active t) (work : List Program) :
    (Interface.io (algebras trees) t.call (metadata t (wordRead before)) (hostScale t.call.dim)
      work Native.Term.leaf (Native.eval I fusion (scaleWord t.call.dim))).InputBinding
        (LegacyBinding.values t.call (metadata t (wordRead before)) (buffers t before))
        (initial t.call (metadata t (wordRead before)) (Native.injectBuffers t.call (buffers t before))
          (Native.injectState rest)) :=
  LegacyBinding.input _ _ _ _ work _ _ _ (Native.injectState rest)
    (valid t (wordRead before) contract launch)

theorem query_logical {α : Type} [Inhabited α] (t : TensorCall) (words : Nat → Word)
    (before : Cells α) (contract : InputContract t words) (launch : active t)
    (b : Fin t.call.batches) (h : Fin t.call.heads) (d : Fin t.call.dim) :
    Interface.query (LegacyBinding.values t.call (metadata t words) (buffers t before)) b h d =
      logical before t.q (queryIndex t b.val h.val d.val) :=
  query_tensor t words before contract launch b.val h.val b.isLt h.isLt d

/-- The gather reads the original shared physical reader. No per-view copy or
new separation requirement is inserted for read-only aliases. -/
theorem key_logical {α : Type} [Inhabited α] (t : TensorCall) (words : Nat → Word)
    (before : Cells α) (contract : InputContract t words) (launch : active t)
    (b : Fin t.call.batches) (h : Fin t.call.heads) (n : Nat)
    (hn : n<(metadata t words).len b.val) (d : Fin t.call.dim) :
    Reference.keyAt t.call (LegacyBinding.values t.call (metadata t words) (buffers t before))
      ((metadata t words).page b.val n) (h.val/(t.call.heads/t.call.kvHeads)) d =
      logical before t.k (keyIndex t words b.val h.val n d.val) := by
  have v := valid t words contract launch
  have hp := (v.pages b n hn).2
  have hg := grouped_head t.call (metadata t words) v h.val h.isLt
  simpa [Reference.keyAt,hp,hg,Interface.key,LegacyBinding.values,Reference.keyAddress,Whole.key] using
    key_tensor t words before contract launch b.val h.val n b.isLt h.isLt hn d

#axiomsClean real_inputs
#axiomsClean native_inputs
#axiomsClean query_logical
#axiomsClean key_logical
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.PhysicalIOBinding
