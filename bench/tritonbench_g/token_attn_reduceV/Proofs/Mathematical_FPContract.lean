import bench.tritonbench_g.token_attn_reduceV.Proofs.Mathematical_FPComplete
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
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

#axiomsClean reference_readback
#axiomsClean reference_frame
end
end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
