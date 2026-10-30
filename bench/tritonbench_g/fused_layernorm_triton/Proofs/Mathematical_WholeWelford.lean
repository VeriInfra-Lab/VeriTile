import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_FPWelfordBlock

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural
open FPStages
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Subsequent source loop iteration. The first four declarations supply
parser register shapes only; they are excluded from the emitted body. -/
def nextChunk (X : RegionName) (D B k : Nat) : List ComputeStmt :=
  ((triton {
    tl.region X = tl.float32
    x0 = tl.program_id(0) + tl.arange(0, $(1))[:, None]
    tmp3_mean = tl.zeros([$(1), $(B)], tl.float32)
    tmp3_m2 = tl.zeros([$(1), $(B)], tl.float32)
    tmp3_weight = tl.zeros([$(1), $(B)], tl.float32)
    roffset = $(k * B)
    rbase = tl.arange(0, $(B))[None, :]
    rindex = roffset + rbase
    rmask = rindex < $(D)
    r1 = rindex
    tmp0 = tl.load(X + (r1 + ($(D) * x0)), mask=rmask).to(tl.float32)
    tmp1 = (tmp0).to(tl.float32)
    tmp2, _ = tl.broadcast(tmp1, tmp3_mean)
    delta = tmp2 - tmp3_mean
    tmp3_weight_next = tmp3_weight + 1.0
    tmp3_mean_next = tmp3_mean + delta / tmp3_weight_next
    delta2 = tmp2 - tmp3_mean_next
    tmp3_m2_next = tmp3_m2 + delta * delta2
    tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
    tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
    tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
  } : ComputeKernel).surfaceBody).drop 4

def advance {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B k : Nat) (previous : Fin B → WelfordTriple α) :
    Fin B → WelfordTriple α := fun i =>
  if k * B + i.val < D then
    reduceStep M Bool.false
      (M.cast (some .fp32) .real .real
        (M.cast (some .fp32) .real .real
          ((s.mem X (k * B + i.val + D * s.pids 0)).read .real))) (previous i)
  else previous i

theorem next_chunk_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s t : State α) (X : RegionName) (D B k : Nat)
    (previous : Fin B → WelfordTriple α) (h : WelfordLanesReady s t previous) :
    ∃ final, run (withEquality M equal32) (nextChunk X D B k) t = some final ∧
      WelfordLanesReady s final (advance M s X D B k previous) := by
  simp [nextChunk, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
    numericLt, withEquality, Region.cast, ofFloat, toFloat, h.mean, h.m2,
    h.weight, h.row, h.mem]
  constructor <;> simp [State.setReg, advance, reduceStep, bop, ofFloat, toFloat,
    Fin.eq_zero, h.mem, h.pids, h.undef, h.row]
  all_goals first | rfl | (funext i; first | (split <;> simp_all [Fin.eq_zero]) | simp [bop])
  all_goals norm_num

/-- Remaining iterations are the finite source range, in source order. -/
def chunkCode (X : RegionName) (D B : Nat) : Nat → Nat → List ComputeStmt
  | _, 0 => []
  | start, count + 1 => nextChunk X D B start ++ chunkCode X D B (start + 1) count

def laneRecurrence {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B : Nat) : Nat → Nat → (Fin B → WelfordTriple α) →
      (Fin B → WelfordTriple α)
  | _, 0, previous => previous
  | start, count + 1, previous =>
    laneRecurrence M s X D B (start + 1) count (advance M s X D B start previous)

theorem chunks_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s t : State α) (X : RegionName) (D B start count : Nat)
    (previous : Fin B → WelfordTriple α) (h : WelfordLanesReady s t previous) :
    ∃ final, run (withEquality M equal32) (chunkCode X D B start count) t = some final ∧
      WelfordLanesReady s final (laneRecurrence M s X D B start count previous) := by
  induction count generalizing start t previous with
  | zero => exact ⟨t, by simp [chunkCode, run], h⟩
  | succ count ih =>
    obtain ⟨u, hu, hp⟩ := next_chunk_run M equal32 s t X D B start previous h
    obtain ⟨final, hf, hr⟩ := ih u (start + 1) _ hp
    exact ⟨final, by simp [chunkCode, run_append, hu, hf], hr⟩

def blockCount (D B : Nat) : Nat := (D + B - 1) / B

def allLaneValues {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X : RegionName) (D B : Nat) : Fin B → WelfordTriple α :=
  laneRecurrence M s X D B 1 (blockCount D B - 1) (firstLaneValues M s X D B)

/-- The complete first loop, helper tree and ordered statistics stores. -/
def statisticsKernel (X Mean Rstd : RegionName) (S D B : Nat)
    (tree : ReductionTree B) : ComputeKernel :=
  .mk [X] [Mean, Rstd]
    (welfordPrefix X S D B ++ chunkCode X D B 1 (blockCount D B - 1) ++
      welfordTupleCode tree ++ welfordSuffix Rstd Mean D)

theorem whole_statistics {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) :
    ∃ final, FP.Structural.exec (withEquality M equal32)
      (statisticsKernel X Mean Rstd S D B tree) s = some final ∧
      final.mem = (statisticsMemory M s Mean Rstd D
        (treeValue M equal32 (allLaneValues M s X D B) tree)).mem := by
  obtain ⟨p, hp, hpr⟩ := welford_prefix_run M equal32 s X S D B
  obtain ⟨q, hq, hqr⟩ := chunks_run M equal32 s p X D B 1 (blockCount D B - 1) _ hpr
  obtain ⟨u, hu, hur⟩ := welford_tuple_run M equal32 s q _ hqr tree
  obtain ⟨final, hf, hm⟩ := welford_suffix_run M equal32 s u Mean Rstd D _ hur
  refine ⟨final, ?_, hm⟩
  simp only [statisticsKernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    run_append, hp, hq, hu, hf, bind, pure, Option.bind]

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

theorem whole_statistics_ready {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X Mean Rstd : RegionName)
    (S D B : Nat) (tree : ReductionTree B) :
    ∃ final, FP.Structural.exec (withEquality M equal32)
      (statisticsKernel X Mean Rstd S D B tree) s = some final ∧
      StatisticsReady s final Mean Rstd D M
        (treeValue M equal32 (allLaneValues M s X D B) tree) := by
  obtain ⟨p, hp, hpr⟩ := welford_prefix_run M equal32 s X S D B
  obtain ⟨q, hq, hqr⟩ := chunks_run M equal32 s p X D B 1 (blockCount D B - 1) _ hpr
  obtain ⟨u, hu, hur⟩ := welford_tuple_run M equal32 s q _ hqr tree
  obtain ⟨final, hf, hm⟩ := statistics_suffix_ready M equal32 s u Mean Rstd D _ hur
  refine ⟨final, ?_, hm⟩
  simp only [statisticsKernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    run_append, hp, hq, hu, hf, bind, pure, Option.bind]

#axiomsClean whole_statistics_ready
#axiomsClean next_chunk_run
#axiomsClean chunks_run
#axiomsClean whole_statistics
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
