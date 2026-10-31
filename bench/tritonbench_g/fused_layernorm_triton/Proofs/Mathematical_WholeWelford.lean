import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_FPWelfordBlock

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural
open FPStages
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def advance {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B k : Nat) (previous : Fin B → WelfordTriple α) :
    Fin B → WelfordTriple α := fun i =>
  if k * B + i.val < D then
    reduceStep M Bool.false
      (M.cast (some .fp32) .real .real
        (M.cast (some .fp32) .real .real
          ((s.mem X (k * B + i.val + D * s.pids 0)).read .real))) (previous i)
  else previous i

def laneRecurrence {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B : Nat) : Nat → Nat → (Fin B → WelfordTriple α) →
      (Fin B → WelfordTriple α)
  | _, 0, previous => previous
  | start, count + 1, previous =>
    laneRecurrence M s X D B (start + 1) count (advance M s X D B start previous)

def blockCount (D B : Nat) : Nat := (D + B - 1) / B

def allLaneValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B : Nat) : Fin B → WelfordTriple α :=
  laneRecurrence M s X D B 1 (blockCount D B - 1) (firstLaneValues M s X D B)

structure StatisticsReady {α : Type} (s t : State α) (Mean Rstd : RegionName)
    (D : Nat) (M : Algebra α) (v : WelfordTriple α) : Prop where
  mem : t.mem = (statisticsMemory M s Mean Rstd D v).mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  mean : t.regs .real [1,1] "tmp3" = some (fun _ => v.mean)
  rstd : t.regs .real [1,1] "tmp10" = some (fun _ => rstdValue M D v)
  row : t.regs .nat [1,1] "x0" = some (fun _ => s.pids 0)

theorem statistics_suffix_ready {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s t : State α) (Mean Rstd : RegionName)
    (D : Nat) (v : WelfordTriple α) (h : WelfordTupleReady s t v) :
    ∃ final, run (withEquality M equal32) (welfordSuffix Rstd Mean D) t = some final ∧
      StatisticsReady s final Mean Rstd D M v := by
  simp [welfordSuffix, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, store,
    withEquality, Region.cast, ofFloat, toFloat, h.mean, h.m2, h.weight, h.row,
    TileShape.allIndices, List.finRange_succ, Fin.eq_zero]
  constructor <;> simp [State.setReg, State.write, statisticsMemory, rstdValue,
    h.mem, h.pids, h.undef, h.row, Fin.eq_zero]
  all_goals funext i; simp [bop]

end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
