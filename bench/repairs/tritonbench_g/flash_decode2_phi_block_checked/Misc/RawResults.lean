import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.RawInterface

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- The complete masked source grid computes the independent unshifted
weighted-summary formula from raw finite summary reads; all other cells frame. -/
theorem real (B H S D Q BD : Nat) (width : D ≤ BD) (order : List Nat)
    (all : order.Perm (List.range (B*H))) (before : State (Option ℝ))
    (x : Nat → Nat → ℝ) (v : Nat → Nat → TileIndex [BD] → ℝ)
    (positive : ∀ row,row< B*H → 0<count (Grid.setup B H row before) Q)
    (logs : ∀ row,row< B*H → ∀ j,j<count (Grid.setup B H row before) Q →
      logvalue Reference.ideal (Grid.setup B H row before) H S j=some (x row j))
    (mids : ∀ row,row< B*H → ∀ j,j<count (Grid.setup B H row before) Q → ∀ k,
      FlashDecode2.value Reference.ideal (Grid.setup B H row before) Bool.true H S D j k=some (v row j k)) :
    Spec.Real (∃ after,Grid.source Reference.ideal Bool.true B H S D Q BD order before=some after ∧
      (∀ row d,row < B*H → ∀ hd : d < D,
        after.mem "O" (row*D+d)=.mk .real
          (some (Reference.target (x row) (v row) (count (Grid.setup B H row before) Q)
            (⟨d,lt_of_lt_of_le hd width⟩,PUnit.unit)))) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ B*H*D ≤ i → after.mem r i=before.mem r i)) := by
  obtain ⟨after,run,values,frame⟩ := Raw.complete Reference.ideal B H S D Q BD order all before
  refine ⟨after,run,?_,frame⟩
  intro row d hr hd
  let k : TileIndex [BD] := (⟨d,lt_of_lt_of_le hd width⟩,PUnit.unit)
  rw [values row k hr hd]
  exact congrArg (Cell.mk .real) (Reference.source_correct (Grid.setup B H row before)
    Bool.true H S D Q BD (x row) (v row) (positive row hr) (logs row hr) (mids row hr) k)

theorem empty {α : Type} [Inhabited α] (A : Algebra α) (B H S D Q BD : Nat)
    (before : State α) (h : B=0 ∨ H=0) :
    Grid.source A Bool.true B H S D Q BD (List.range (B*H)) before=some before := by
  rcases h with h|h <;> simp [h,Grid.source]

theorem source (H S D Q BD : Nat) : FlashDecode2.kernel Bool.true H S D Q BD=phi H S D Q BD := rfl

#axiomsClean real
#axiomsClean empty
#specNonCircular Reference.target avoiding [Grid.source,FP.Structural.exec,values,Reference.recurrence]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked


namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- Complete masked native source grid and independent ordered scalar graph,
under the original instruction meanings and compiler fusion/order choices. -/
theorem native (p : KernelNative.Ops) (schedule : Native.Schedule) (B H S D Q BD : Nat) (width : D ≤ BD)
    (order : List Nat) (all : order.Perm (List.range (B*H))) (before : State Native.Term) :
    ∃ after,Grid.source Native.algebra Bool.true B H S D Q BD order before=some after ∧
      (∀ row d,row < B*H → d < D →
        Native.eval (KernelNative.lower p) schedule ((after.mem "O" (row*D+d)).read .real)=
          Native.eval (KernelNative.lower p) schedule (Raw.nativeTerm before B H S D Q row d)) ∧
      (∀ r i,r ≠ ("O" : RegionName) ∨ B*H*D ≤ i → after.mem r i=before.mem r i) := by
  obtain ⟨after,run,values,frame⟩ := Raw.complete Native.algebra B H S D Q BD order all before
  refine ⟨after,run,?_,frame⟩
  intro row d hr hd
  let k : TileIndex [BD] := (⟨d,lt_of_lt_of_le hd width⟩,PUnit.unit)
  rw [values row k hr hd,Cell.read_same,Raw.native_value]

#axiomsClean native
#specNonCircular Raw.nativeTerm avoiding [Grid.source,FP.Structural.exec,values]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked
