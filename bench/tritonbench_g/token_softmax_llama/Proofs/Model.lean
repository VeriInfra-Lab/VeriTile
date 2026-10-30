import bench.tritonbench_g.token_softmax_llama.Kernels
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/- Source Parameters and kernel retain their names in Kernels.lean. -/

def execution {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (hwidth : 0  <  p.width) : {t : State α // FP.Structural.exec M (kernel d p) s = some t} := by
  rcases p with ⟨x,o,st,len,ss,ls,xh,xt,oh,ot,w⟩
  cases d
  all_goals
    refine ⟨?_, ?_⟩; swap
    simp [kernel, source, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric, bop, numericLt,
      natLt, ofFloat, toFloat, Region.cast, Option.bind, FloatDType.toTileDType,
      FloatDType.numericDType, ComputeDType.eraseDType, TileShape.axisDim, hwidth]
    try rfl

def startIndex {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.start) (s.pids 0*p.startStride)).read .nat

def count {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.length) (s.pids 0*p.lengthStride)).read .nat

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : TileIndex [p.width]) : Nat :=
  s.pids 1 * p.outputHeadStride + (startIndex p s + i.1.val) * p.outputTokenStride

/-- Eager padded lanes, including the conversion of negative infinity, are
retained before maximum, exp, sum and division. No real exp or sum is used. -/
def inputRow {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) : α :=
  M.cast (some .fp32) d .real
    (if i.1.val  <  count p s then toFloat d ((s.mem p.input
      (s.pids 1 * p.inputHeadStride + (startIndex p s + i.1.val) * p.inputTokenStride)).read d.toTileDType)
     else M.cast (some .fp32) .real d M.negInf)

def numerators {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) : TileIndex [p.width] → α :=
  let row := inputRow M d p s
  let maximum := M.reduceMax (some .fp32) ⟨0, by simp⟩ Bool.false row PUnit.unit
  fun i => M.unary (some .fp32) .exp (M.binary (some .fp32) .real .sub (row i) maximum)

def outputValues {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) : TileIndex [p.width] → α :=
  let ns := numerators M d p s
  let total := M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false ns PUnit.unit
  fun i => M.cast none .real d (M.binary (some .fp32) .real .div (ns i) total)

def reference {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if i.1.val  <  count p s then t.write p.output (outputOffset p s i)
      (.mk d.toTileDType (ofFloat d (outputValues M d p s i))) else t) s

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (d : FloatDType)
    (p : Parameters) (s : State α) (hwidth : 0  <  p.width) : (execution M d p s hwidth).val.mem = (reference M d p s).mem := by
  rcases p with ⟨x,o,st,len,ss,ls,xh,xt,oh,ot,w⟩
  cases d <;>
    simp [execution, reference, outputValues, numerators, inputRow, outputOffset,
      startIndex, count, fold_writes_mem, State.setReg, ofFloat, toFloat, Region.cast,
      FloatDType.toTileDType, FloatDType.numericDType, bop, Function.comp_def,
      Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals rfl

#axiomsClean execution
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
