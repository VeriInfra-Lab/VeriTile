import bench.tritonbench_g.token_softmax_bloom.Proofs.MixedSource
import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedIO
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.MixedTyped
open VeriTile.Triton FP.Structural TokenSoftmaxBloom.Typed
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- Selected ordinary floating storage routes; input and output vary independently. -/
inductive Storage where
  | half | bfloat | single
  deriving DecidableEq, Repr

def dtype : Storage → FloatDType
  | .half => .fp16 | .bfloat => .bf16 | .single => .fp32

def kernel (d out : Storage) (p : Parameters) := source (dtype d) (dtype out) p.input p.start p.length p.output
  p.inputHeadStride p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width

def execution {α : Type} [Inhabited α] (M : Algebra α) (d out : Storage)
    (p : Parameters) (s : State α) (hwidth : 0 < p.width) : {t : State α // FP.Structural.exec M (kernel d out p) s = some t} := by
  rcases p with ⟨x,o,st,len,xh,xt,oh,ot,w⟩
  cases d <;> cases out
  all_goals
    refine ⟨?_, ?_⟩; swap
    simp [kernel, source, FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric, bop, numericLt,
      natLt, ofFloat, toFloat, Region.cast, Option.bind, FloatDType.toTileDType,
      FloatDType.numericDType, dtype, ComputeDType.eraseDType, TileShape.axisDim, hwidth]
    try rfl

/-- Independent initial-state formula, preserving the fp32 reduction order and
only changing the final storage conversion with the destination selector. -/
def outputValues {α : Type} [Inhabited α] (M : Algebra α) (d out : Storage)
    (p : Parameters) (s : State α) : TileIndex [p.width] → α :=
  let ns := numerators M (dtype d) p s
  let total := M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false ns PUnit.unit
  fun i => M.cast none .real (dtype out) (M.binary (some .fp32) .real .div (ns i) total)

def reference {α : Type} [Inhabited α] (M : Algebra α) (d out : Storage)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl (fun t i =>
    if i.1.val < count p s then t.write p.output (outputOffset p s i)
      (.mk (dtype out).toTileDType (ofFloat (dtype out) (outputValues M d out p s i))) else t) s

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (d out : Storage)
    (p : Parameters) (s : State α) (hwidth : 0 < p.width) : (execution M d out p s hwidth).val.mem = (reference M d out p s).mem := by
  rcases p with ⟨x,o,st,len,xh,xt,oh,ot,w⟩
  cases d <;> cases out <;>
    simp [execution, reference, outputValues, numerators, inputRow, outputOffset,
      startIndex, count, fold_writes_mem, State.setReg, ofFloat, toFloat, Region.cast,
      FloatDType.toTileDType, FloatDType.numericDType, dtype, bop, Function.comp_def,
      Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals rfl

theorem homogeneous_kernel (d : Storage) (p : Parameters) :
    kernel d d p = Typed.kernel (dtype d) p := by
  cases d <;> rfl

#specNonCircular reference avoiding [kernel, source, FP.Structural.exec, run, step]
#axiomsClean homogeneous_kernel
#axiomsClean execution
#axiomsClean execution_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.MixedTyped
