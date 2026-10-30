import bench.repairs.tritonbench_g.rope_backward_transform.Contract

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelInterface
open VeriTile.Triton FP.Structural
open KernelContract (launchRows launched nativePrimitives)
noncomputable section

/-- Exact raw regional footprint of both Q and K rotation pairs. -/
def Written (r : Request) (name : RegionName) (offset : Nat) : Prop :=
  ∃ key row head col, name=region key ∧ row<r.rows ∧ head<r.heads key ∧
    col<2*r.half ∧ offset=r.offset key row head col

/-- The complete row grid, including odd tails and every empty/no-pair case. -/
theorem complete {α : Type} [Inhabited α] (op : Operations α)
    (r : Request) (m : Source.Memory α) :
    ∃ final, GridExec op.backend r (launchRows r) m final ∧
      (∀ key row head col, row<r.rows → head<r.heads key → col<r.dimension →
        (final (region key) (r.offset key row head col)).read .fp32 =
          scalarResult op r m key row head col) ∧
      (∀ name offset, ¬ Written r name offset → final name offset=m name offset) := by
  refine ⟨grid op.backend r m (launchRows r), grid_execution _ _ _ _, ?_, ?_⟩
  · intro key row head col hr hh hc
    by_cases on : launched r
    · rw [launchRows, if_pos on]
      exact grid_readback op r m _ List.nodup_range row head col key (List.mem_range.mpr hr) hh hc
    · have half : r.half=0 := by
        by_contra positive
        apply on
        refine ⟨by omega, by omega, ?_⟩
        cases key <;> simp only [Request.heads, Bool.false_eq_true, ↓reduceIte] at hh
        · exact Or.inl (by omega)
        · exact Or.inr (by omega)
      simp only [launchRows, if_neg on, grid, scalarResult, half, Nat.mul_zero,
        Nat.not_lt_zero, ↓reduceIte]
  · intro name offset outside
    apply grid_frame
    intro row member key side
    have hr : row<r.rows := by
      unfold launchRows at member
      split at member
      · exact List.mem_range.mp member
      · simp at member
    by_cases same : name=region key
    · right
      intro i active equal
      apply outside
      obtain ⟨hh,hj⟩ := (active_iff _ _ i).mp active
      refine ⟨key,row,i.1.val,i.2.1.val+if side then r.half else 0,same,hr,hh,?_,?_⟩
      · cases side <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> omega
      · simpa only [address,address_coordinate] using equal
    · exact Or.inl same

/-- Independent fixed-coefficient transpose on raw initial values. -/
def realValue (r : Request) (m : Source.Memory ℝ) (key : Bool) (row head col : Nat) : ℝ :=
  if col<2*r.half then
    let j := if col<r.half then col else col-r.half
    let pair := Spec.pair
      ((m (region key) (r.offset key row head j)).read .fp32)
      ((m (region key) (r.offset key row head (j+r.half))).read .fp32)
      ((m "C" (row%r.sequence*r.cosRowStride+j*r.cosColStride)).read .fp32)
      ((m "S" (row%r.sequence*r.sinRowStride+j*r.sinColStride)).read .fp32)
    if col<r.half then pair.1 else pair.2
  else (m (region key) (r.offset key row head col)).read .fp32

theorem real_binding (r : Request) (m : Source.Memory ℝ) (key : Bool) (row head col : Nat) :
    scalarResult ideal r m key row head col=realValue r m key row head col := by
  unfold scalarResult realValue
  split
  · dsimp only
    rw [ideal_pair]
    by_cases low : col<r.half <;> simp [low,show (r.half≤col)↔¬col<r.half from by omega]
  · rfl

#axiomsClean complete
#axiomsClean real_binding
#specNonCircular realValue avoiding [Source.kernel, grid, scalarResult, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelInterface
