import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeRow

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def atRow {α : Type} (s : State α) (row : Nat) : State α :=
  { s with pids := fun dim => if dim = 0 then row else 0 }

/-- A generic dispatch fold. The theorem below allows any order and repetitions;
completion of a grid means its list contains every launched row. -/
def runRows {α : Type} [Inhabited α] (M : Algebra α) (kernel : ComputeKernel) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | row :: rows, s => do
      let t ← FP.Structural.exec M kernel (atRow s row)
      runRows M kernel rows t

theorem lane_recurrence_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s t : State α) (X : RegionName) (D B start count : Nat)
    (previous : Fin B → WelfordTriple α) (hm : s.mem X = t.mem X) (hp : s.pids 0 = t.pids 0) :
    laneRecurrence M s X D B start count previous = laneRecurrence M t X D B start count previous := by
  induction count generalizing start previous with
  | zero => rfl
  | succ n ih =>
    simp only [laneRecurrence]
    rw [ih]
    congr 1
    funext i
    simp only [advance, hm, hp]

theorem row_value_congr {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X : RegionName) (D B row : Nat) (tree : ReductionTree B)
    (hm : s.mem X = t.mem X) :
    rowValue M eq32 (atRow s row) X D B tree = rowValue M eq32 (atRow t row) X D B tree := by
  unfold rowValue allLaneValues
  rw [lane_recurrence_congr M (atRow s row) (atRow t row) X D B 1 (blockCount D B - 1) _ hm rfl]
  have hf : firstLaneValues M (atRow s row) X D B = firstLaneValues M (atRow t row) X D B := by
    funext i
    simp [firstLaneValues, atRow, hm]
  rw [hf]

def rowOutput {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s : State α) (X W Bias : RegionName) (D B row col : Nat) (tree : ReductionTree B) : α :=
  normalizeValue M s X W Bias D row col (rowValue M eq32 (atRow s row) X D B tree).mean
    (rstdValue M D (rowValue M eq32 (atRow s row) X D B tree))

structure GridEffect {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (s t : State α) (X W Bias Y Mean Rstd : RegionName) (D B : Nat)
    (tree : ReductionTree B) (rows : List Nat) : Prop where
  mean : ∀ row ∈ rows, t.mem Mean row = .mk .real (rowValue M eq32 (atRow s row) X D B tree).mean
  rstd : ∀ row ∈ rows, t.mem Rstd row = .mk .real (rstdValue M D (rowValue M eq32 (atRow s row) X D B tree))
  output : ∀ row ∈ rows, ∀ col, col < D → t.mem Y (col + D * row) =
    .mk .bf16 (rowOutput M eq32 s X W Bias D B row col tree)
  frame : ∀ q off, (∀ row ∈ rows, rowMiss Y Mean Rstd D row q off) → t.mem q off = s.mem q off

end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
