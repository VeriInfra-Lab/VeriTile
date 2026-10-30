import bench.tritonbench_g.matmul_persistent_triton.Registers
import bench.tritonbench_g.matmul_persistent_triton.PrefixExecution

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private theorem min_lt_form (a b : Nat) : min a b = if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

theorem ceil_exact (n d : Nat) (width : d=64 ∨ d=128 ∨ d=256) :
    ceil n d=RealMath.blocks n d := by
  rw [← source_ceil n d width]
  unfold ceil
  by_cases h : n%d=0 <;> simp [h]

variable {α : Type} [Inhabited α]

theorem prelude_invariant (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    Registers p (s.pids 0) (preludeExecution P p s).val ∧
    (preludeExecution P p s).val.mem=s.mem := by
  have hM : RealMath.blocks p.problem.M 128=ceil p.problem.M 128 := (ceil_exact _ _ (by omega)).symm
  have hN : RealMath.blocks p.problem.N 256=ceil p.problem.N 256 := (ceil_exact _ _ (by omega)).symm
  have hK : RealMath.blocks p.problem.K p.problem.profile.blockK=ceil p.problem.K p.problem.profile.blockK :=
    (ceil_exact _ _ (by cases p.problem.profile <;> simp [Storage.blockK])).symm
  by_cases hm : p.problem.M%128=0 <;>
      by_cases hn : p.problem.N%256=0 <;>
      by_cases hk : p.problem.K%p.problem.profile.blockK=0 <;>
      by_cases hs : (s.pids 0 : Int) < ((ceil p.problem.M 128 * ceil p.problem.N 256)%p.problem.SMS : Nat)
  all_goals
    have hsN := hs
    norm_cast at hsN
    constructor
    · constructor
      all_goals
        simp only [preludeExecution,hm,hn,hk,hs,↓reduceDIte]
        simp_all [Problem.rows,Problem.columns,Problem.kTiles,Problem.work,Problem.tiles,
          Problem.group,Schedule.tilesFor,hM,hN,hK,ceil,FP.Structural.bop,
          Broadcast.leftIndex,Broadcast.rightIndex,min_lt_form]
      all_goals
        funext i
        simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
        all_goals assumption
    · simp only [preludeExecution,hm,hn,hk,hs,↓reduceDIte]
      rfl

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
