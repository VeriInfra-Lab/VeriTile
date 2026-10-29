import bench.tritonbench_g.lora_expand_gemv.Proofs.FPReference

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

attribute [local irreducible] castWord mulWord addWord treeValue

def preludeExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    {t : State Word // run (backend rest cfg plans) (prelude profile out p) s = some t} := by
  cases profile <;> cases out
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [prelude, source, source_halfHalf, source_halfBfloat, source_bfloatHalf,
      source_bfloatBfloat, outputType, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind, backend]
    rfl

theorem preludeInvariant (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    Invariant cfg plans profile out p s 0 (preludeExecution rest cfg plans profile out p s).val := by
  cases profile <;> cases out
  all_goals constructor
  all_goals simp [preludeExecution, backend, formatOf, aValue, aOffset, bOffset, selected,
    weightBase, outputBase, inputType, weightType, ofFloat, toFloat,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    show (0.0 : ℝ) = 0 by norm_num]
  all_goals
    funext i
    simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]

/-- Executed body state, with the initial cached input tile retained in its
register and only the current row block written to memory. -/
def bodyResult (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) (profile : InputProfile)
    (out : OutputFormat) (p : Parameters) (initial current : State Word) (n : Nat) : State Word :=
  let s1 := current.setReg "n" .nat [] (fun _ => n)
  let s2 := s1.setReg "current_n" .nat [p.blockRows] (fun i => n+i.1.val)
  let s3 := s2.setReg "tiled_b" (weightType profile).toTileDType [p.blockRows,p.blockInner]
    (fun i => ofFloat (weightType profile) (bValue profile p initial (n+i.1.val) i.2.1))
  let s4 := s3.setReg "accumulator" (productType profile).toTileDType [p.blockRows]
    (fun i => ofFloat (productType profile) (accumulated cfg plans profile p initial (n+i.1.val)))
  writeBlock cfg plans profile out p initial s4 n

def bodyExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial current : State Word)
    (n : Nat) (h : Invariant cfg plans profile out p initial n current) :
    {t : State Word // run (backend rest cfg plans) (body profile out p)
      (current.setReg "n" .nat [] (fun _ => n)) = some t} := by
  refine ⟨bodyResult cfg plans profile out p initial current n, ?_⟩
  cases profile <;> cases out
  all_goals
    have hav := h.av
    simp only [inputType, FloatDType.toTileDType, ofFloat] at hav
    simp [body, source, source_halfHalf, source_halfBfloat, source_bfloatHalf,
      source_bfloatBfloat, outputType, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
      h.offk, h.offn, hav, h.bp, h.cp, h.weights,
      inputType, weightType, productType, backend, formatOf, typedSum,
      bodyResult, writeBlock, outputCell, expected, accumulated, product,
      aValue, bValue, outputOffset, outputBase, bOffset, weightBase,
      productFormat, inputFormat, weightFormat, sumValues,
      Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.insertAxisIndex, TileShape.dropInsertedIndex, TileShape.eraseAxis,
      show (0.0 : ℝ) = 0 by norm_num]
    simp only [product_function, inputFormat, weightFormat, productFormat, castWord_same]
    dsimp only [product, inputFormat, weightFormat, productFormat, aValue, bValue,
      inputType, weightType, toFloat, bOffset, weightBase, FP.Structural.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, TileShape.axisDim]
    try simp only [castWord_same, castWord_zero]
    try rfl

#axiomsClean preludeExecution
#axiomsClean preludeInvariant
#axiomsClean bodyExecution
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
