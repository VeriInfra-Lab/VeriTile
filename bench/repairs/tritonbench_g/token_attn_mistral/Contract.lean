import bench.repairs.tritonbench_g.token_attn_mistral.Execution
namespace VeriTile.Bench.Repairs.TokenAttnMistral
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 22000
set_option linter.unusedSimpArgs false

theorem fold_write_memory {α ι : Type} (xs : List ι) (region : ι → RegionName)
    (offset : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => t.write (region i) (offset i) (v i)) s).mem =
    xs.foldl (fun mem i => fun r o => if r = region i ∧ o = offset i then v i else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih =>
    rw [List.foldl_cons, ih]
    rfl

theorem postlude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (h : Ready p s t (accumulated M R p s)) :
    ∃ u, run (withSum M R) (postlude p) t = some u ∧ u.mem = (reference M R p s).mem := by
  simp [postlude, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, store, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, Region.cast, withSum, h.mem, h.pids, h.undef, h.batch, h.head,
    h.offsN, h.offsD, h.length, h.prob, h.gather, h.value, h.accumulator]
  simp only [reference, fold_write_memory, State.setReg_mem, h.mem]
  rfl

theorem source_split (p : Parameters) : (kernel p).surfaceBody = prelude p ++
    [ComputeStmt.forRangeDyn "start_n" (Op.constNat 0)
      (Op.ref .nat [] "cur_att_seq_len") (Op.constNat p.block) (body p)] ++ postlude p := by
  rfl

theorem complete_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (withSum M R) (kernel p) s = some t ∧
      t.mem = (reference M R p s).mem := by
  obtain ⟨a,ha,hr⟩ := prelude_run M R p s
  obtain ⟨b,hb,hs⟩ := range_run M R p s a 0 (count p s) (initial M p) hr
  obtain ⟨c,hc,hm⟩ := postlude_run M R p s b hs
  refine ⟨c,?_,hm⟩
  change run (withSum M R) (kernel p).surfaceBody s = some c
  rw [source_split]
  simp only [run_append, ha, run, List.foldlM_cons, List.foldlM_nil,
    bind, pure, Option.bind]
  simp only [step, evalOp_unfold, hr.length, bind, pure, Option.bind, hb, hc]

/-- Full output footprint: the final source store is unmasked. -/
def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (off : Nat) :=
  r ≠ p.output ∨ ∀ i : TileIndex [p.width], off ≠ outputOffset p s i

theorem reference_readback {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (hinj : Function.Injective (outputOffset p s))
    (i : TileIndex [p.width]) :
    (reference M R p s).mem p.output (outputOffset p s i) = .mk .fp32 (outputValues M R p s i) := by
  exact State.scatter_readback p.output (outputOffset p s)
    (fun j => Cell.mk .fp32 (outputValues M R p s j)) s i hinj

theorem reference_frame {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (r : RegionName) (off : Nat) (h : Outside p s r off) :
    (reference M R p s).mem r off = s.mem r off := by
  apply State.scatter_frame
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i _ => h i)

#axiomsClean postlude_run
#axiomsClean source_split
#axiomsClean complete_run
#axiomsClean reference_readback
#axiomsClean reference_frame
end
end VeriTile.Bench.Repairs.TokenAttnMistral
