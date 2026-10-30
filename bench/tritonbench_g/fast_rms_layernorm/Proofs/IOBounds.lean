import bench.tritonbench_g.fast_rms_layernorm.IO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IOBounds
open VeriTile Triton FP.Structural NativeGraph
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Capacity (h : Call) (backward : Bool) (bounds : RegionName → Nat) : Prop where
  x : h.cells≤bounds "x"
  w : h.cols≤bounds "w"
  r : h.rows≤bounds "r"
  row : h.cells≤bounds (if backward then "dx" else "y")
  dw : backward=Bool.true → h.cols≤bounds "dw"

private theorem capacity {n m : Nat} (inside : ∀ i : Fin n,i.val<m) : n≤m := by
  by_contra outside
  have h:=inside ⟨m,by omega⟩
  change m<m at h
  omega

theorem real (h : Call) (backward : Bool) (vals : (RealIO.program h backward).Ctx)
    (s : BlockState) (bounds : RegionName → Nat)
    (bounded : (RealIO.program h backward).InBounds bounds s vals) : Capacity h backward bounds := by
  dsimp only [PreparedAlgorithmProgramIO.InBounds,AlgorithmMemoryIO.InBounds,RealIO.program,RealIO.memory] at bounded
  refine ⟨capacity ?_,capacity ?_,capacity ?_,capacity ?_,?_⟩
  · intro i
    exact bounded.1 0 i (Or.inl (by decide))
  · intro i
    have got:=bounded.1 1 i (Or.inl (by decide))
    exact got
  · intro i
    cases backward
    · exact bounded.2.1 1 i trivial
    · exact bounded.1 2 i (Or.inr ⟨rfl,Or.inl (by decide)⟩)
  · intro i
    cases backward <;> exact bounded.2.1 0 i trivial
  · intro enabled
    subst backward
    exact capacity (fun i=>bounded.2.1 1 i trivial)

theorem native (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term)
    (vals : (NativeIO.program R h backward epsilon).Inputs) (bounds : RegionName → Nat)
    (bounded : (NativeIO.program R h backward epsilon).InBounds bounds vals) : Capacity h backward bounds := by
  dsimp only [ProgramIO.InBounds,NativeIO.program] at bounded
  refine ⟨capacity ?_,capacity ?_,capacity ?_,capacity ?_,?_⟩
  · intro i
    have got:=bounded.2.1 ⟨0,by simp [Ports.inputs]⟩ (i,()) (Or.inl (by simp))
    exact got
  · intro i
    have got:=bounded.2.1 ⟨1,by simp [Ports.inputs]⟩ (i,()) (Or.inl (by simp))
    exact got
  · intro i
    cases backward
    · have got:=bounded.2.2.1 ⟨1,by simp [Ports.outputs]⟩ (i,()) trivial
      exact got
    · have got:=bounded.2.1 ⟨2,by simp [Ports.inputs]⟩ (i,()) (Or.inr ⟨rfl,Or.inl (by simp)⟩)
      exact got
  · intro i
    cases backward
    all_goals
      have got:=bounded.2.2.1 ⟨0,by simp [Ports.outputs]⟩ (i,()) trivial
      exact got
  · intro enabled
    subst backward
    apply capacity
    intro i
    have got:=bounded.2.2.1 ⟨1,by simp [Ports.outputs]⟩ (i,()) trivial
    exact got

end
end VeriTile.Bench.Repairs.FastRmsLayernorm.IOBounds
