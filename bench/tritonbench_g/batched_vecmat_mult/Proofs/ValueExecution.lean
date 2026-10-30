import bench.tritonbench_g.batched_vecmat_mult.Proofs.ValueModel

namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] FP.add FP.mul treeValue

def prologue (p : Parameters) := (source p).surfaceBody.take 5
def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[5]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ b) => b
  | _ => []
def tail (p : Parameters) := (source p).surfaceBody.drop 6

theorem source_split (p : Parameters) :
    (source p).surfaceBody = prologue p ++
      [.forRangeDyn "k_index" (.constNat 0) (.ref .nat [] "k_blocks") (.constNat 1) (body p)] ++
      tail p := by rfl

def prologueExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) :
    {t : State Word // run (engine rest cfg plans) (prologue p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prologue, source, TypedSource.original, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, numeric, integralFloorDiv, ComputeDType.eraseDType,
    State.setReg, bop]
  rfl

theorem prologueInvariant (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (s : State Word) :
    Invariant cfg plans p s 0 (prologueExecution rest cfg plans p s).val := by
  constructor
  all_goals simp [prologueExecution, accumulated, outputOffset, row, col, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
  all_goals funext i
  all_goals rfl

def bodyExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial current : State Word) (c : Nat)
    (h : Invariant cfg plans p initial c current) :
    {t : State Word // run (engine rest cfg plans) (body p)
      (current.setReg "k_index" .nat [] (fun _ => c)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [body, source, TypedSource.original, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, numeric, integralFloorDiv, ComputeDType.eraseDType,
    State.setReg, bop, h.mi, h.ni, h.offsets, h.carry, h.blocks, h.memory, Region.cast]
  rfl

theorem bodyInvariant (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial current : State Word) (c : Nat)
    (h : Invariant cfg plans p initial c current) :
    Invariant cfg plans p initial (c+1) (bodyExecution rest cfg plans p initial current c h).val := by
  constructor
  all_goals simp [bodyExecution, h.mi, h.ni, h.offsets, h.carry, h.blocks, h.memory,
    accumulated, blockSum, aOffset, bOffset, row, col, sumValues, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex,
    TileShape.insertAxisIndex, TileShape.axisDim, Region.cast]
  all_goals funext i
  all_goals rfl

#axiomsClean source_split
#axiomsClean prologueExecution
#axiomsClean prologueInvariant
#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
