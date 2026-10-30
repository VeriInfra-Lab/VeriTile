import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Forward

namespace VeriTile.Bench.Repairs.KldivOps.Program
open VeriTile.Triton FP.Structural Tile
noncomputable section
set_option maxHeartbeats 6000000

theorem loop_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (start : Nat) (data : Memory α×α) (r : RegionName) (i : Nat)
    (outside : r ≠ p.O ∨ i < row*p.cols ∨ row*p.cols+p.cols ≤ i) :
    (loopModel M p flag red row eps start data).1 r i=data.1 r i := by
  rw [loopModel]
  split
  next active =>
    rw [loop_frame M p flag red row eps (start+p.block) (advance M p flag red row start eps data) r i outside]
    apply chunk_frame M p flag red row start eps (memoryState data.1) r i
    rcases outside with hr | hr
    · exact Or.inl hr
    · right
      intro j hj
      rcases hr with hlo | hhi <;> omega
  next inactive => rfl
termination_by p.cols-start
decreasing_by omega

def Writes (p : Parameters) (red row : Nat) (r : RegionName) (i : Nat) : Prop :=
  r=p.O ∧ if red=0 then row*p.cols ≤ i ∧ i < row*p.cols+p.cols else i=row

theorem row_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (mem : Memory α) (r : RegionName) (i : Nat)
    (outside : ¬Writes p red row r i) :
    rowModel M p flag red row eps mem r i=mem r i := by
  unfold rowModel
  by_cases hz : red=0
  · rw [if_pos hz]
    apply loop_frame
    by_cases hr : r=p.O
    · right
      simp [Writes,hz,hr] at outside
      omega
    · exact Or.inl hr
  · rw [if_neg hz]
    have hstore : ¬(r=p.O ∧ i=row) := by simpa [Writes,hz] using outside
    simp only [Program.writeMemory,if_neg hstore]
    have invariant : ∀ start data,
        (loopModel M p flag red row eps start data).1=data.1 := by
      intro start data
      induction h : p.cols-start using Nat.strong_induction_on generalizing start data with
      | _ n ih =>
        rw [loopModel]
        split
        next active =>
          have hn : p.cols-(start+p.block)<n := by omega
          rw [ih _ hn _ _ rfl]
          simp [advance,chunkMemory,hz,memoryState]
        next => rfl
    rw [invariant]

theorem grid_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : α) (jobs : List Nat) (mem : Memory α) (r : RegionName) (i : Nat)
    (outside : ∀ row∈jobs,¬Writes p red row r i) :
    gridModel M p flag red eps jobs mem r i=mem r i := by
  induction jobs generalizing mem with
  | nil => rfl
  | cons row rest ih =>
    change gridModel M p flag red eps rest (rowModel M p flag red row eps mem) r i=mem r i
    rw [ih _ (by intro x hx; exact outside x (by simp [hx]))]
    exact row_frame M p flag red row eps mem r i (outside row (by simp))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Program
