import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Mathematical_TypedContract
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def combinedOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [11,p.blockSize]) : Nat := outputOffset p s i.1 i.2

def combinedValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [11,p.blockSize]) : Cell α := .mk .real (expected M p s i.1 i.2)

theorem written_scatter {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    written M p s = (TileShape.allIndices [11,p.blockSize]).foldl (fun t i =>
      if combinedOffset p s i < p.outputNumel then
        t.write p.output (combinedOffset p s i) (combinedValue M p s i) else t) s := by
  rw [TileShape.allIndices_cons]
  simp only [List.foldl_flatMap, List.foldl_map]
  rfl

theorem output_injective {α : Type} (p : Parameters) (s : State α) (hstride : 10 < p.outputStride) :
    Function.Injective (combinedOffset p s) := by
  intro i j h
  have hi : i.1.val < p.outputStride := lt_of_lt_of_le i.1.isLt hstride
  have hj : j.1.val < p.outputStride := lt_of_lt_of_le j.1.isLt hstride
  have h' : i.2.1.val * p.outputStride + i.1.val = j.2.1.val * p.outputStride + j.1.val := by
    unfold combinedOffset outputOffset at h
    omega
  have hr : i.2.1.val = j.2.1.val := by
    have hd := congrArg (fun x => x / p.outputStride) h'
    change (i.2.1.val * p.outputStride + i.1.val) / p.outputStride =
      (j.2.1.val * p.outputStride + j.1.val) / p.outputStride at hd
    rw [Nat.mul_comm i.2.1.val, Nat.mul_comm j.2.1.val,
      Nat.mul_add_div (by omega : 0 < p.outputStride),
      Nat.mul_add_div (by omega : 0 < p.outputStride),
      Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hj] at hd
    simpa using hd
  have hc : i.1.val = j.1.val := by
    rw [hr] at h'
    exact Nat.add_left_cancel h'
  exact Prod.ext (Fin.ext hc) (Prod.ext (Fin.ext hr) (Subsingleton.elim _ _))

def Outside {α : Type} (p : Parameters) (s : State α) (r : RegionName) (o : Nat) : Prop :=
  r ≠ p.output ∨ ∀ c i, outputOffset p s c i < p.outputNumel → o ≠ outputOffset p s c i

/-- Every active output channel has all three initial coordinate reads. This
is the same input/output extent coupling as the accepted mathematical stage. -/
def InputCover {α : Type} (p : Parameters) (s : State α) : Prop :=
  ∀ i, outputOffset p s 0 i < p.outputNumel → coordOffset p s i + 2 < p.coordNumel

def rawExpected {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (c : Fin 11) (i : TileIndex [p.blockSize]) : α :=
  channels M (M.fp32Load ((s.mem p.input (coordOffset p s i)).read .real))
    (M.fp32Load ((s.mem p.input (coordOffset p s i + 1)).read .real))
    (M.fp32Load ((s.mem p.input (coordOffset p s i + 2)).read .real)) c

theorem expected_of_cover {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (cover : InputCover p s) (c : Fin 11) (i : TileIndex [p.blockSize])
    (hi : outputOffset p s c i < p.outputNumel) : expected M p s c i = rawExpected M p s c i := by
  have hzero : outputOffset p s 0 i < p.outputNumel := by
    unfold outputOffset at *
    simp only [Fin.val_zero, Nat.add_zero]
    omega
  have hc := cover i hzero
  have h0 : coordOffset p s i < p.coordNumel := by omega
  have h1 : coordOffset p s i + 1 < p.coordNumel := by omega
  simp [expected, rawExpected, loaded, h0, h1, hc]

theorem source_correct {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hstride : 10 < p.outputStride) (cover : InputCover p s) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ c i, outputOffset p s c i < p.outputNumel →
        t.mem p.output (outputOffset p s c i) = .mk .real (rawExpected M p s c i)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o) := by
  refine ⟨(execution M p s).val, (execution M p s).property, ?_, ?_⟩
  · intro c i hi
    rw [execution_memory, written_scatter]
    have hr := State.masked_scatter_readback p.output (combinedOffset p s)
      (combinedValue M p s) (fun i => combinedOffset p s i < p.outputNumel)
      s (c,i) hi (output_injective p s hstride)
    simpa [combinedOffset, combinedValue, expected_of_cover M p s cover c i hi] using hr
  · intro r o h
    rw [execution_memory, written_scatter]
    apply State.masked_scatter_frame
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (fun i _ hi => h i.1 i.2 hi)

#axiomsClean written_scatter
#axiomsClean output_injective
#axiomsClean expected_of_cover
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
