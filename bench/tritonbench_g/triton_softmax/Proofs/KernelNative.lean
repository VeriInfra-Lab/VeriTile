import bench.tritonbench_g.triton_softmax.Proofs.NativeSource
import bench.tritonbench_g.triton_softmax.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.TritonSoftmax.KernelNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def candidateEvaluate {α} [Inhabited α] (M : Algebra α) (x y : Tensor)
    (B p : Nat) (s : State α) : Option (State α) :=
  exec (SoftmaxTriton1.Floating.backend M)
    (indexCommutedKernel y.region x.region x.offset y.offset x.rowStride x.colStride
      y.rowStride y.colStride x.cols B) (Floating.launchState x p s)

theorem execution {α} [Inhabited α] (M : Algebra α) (x y : Tensor)
    (B p : Nat) (s : State α) :
    Floating.evaluate M x y B p s = candidateEvaluate M x y B p s := by
  have rewritten := KernelRewrite.execution (SoftmaxTriton1.Floating.backend M) y.region x.region
    x.offset y.offset x.rowStride x.colStride y.rowStride y.colStride x.cols B
    (Floating.launchState x p s)
  simpa [Floating.evaluate,candidateEvaluate,FP.Structural.exec,softmax_kernel,indexCommutedKernel,
    run,step,evalExpr,FP.Structural.evalOp,store,State.setReg,numeric,natLt,bop,TileShape.axisDim,
    Option.bind,Function.comp_def,ComputeKernel.surfaceBody,evalComputeOp,
    SoftmaxTriton1.Floating.backend,ComputeDType.eraseDType,
    FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,resolvePrecision] using rewritten

def candidate (inst : NativeGraph.Instructions) (V : Nat) (x y : Tensor)
    (B p : Nat) (s : State NativeGraph.Word) : Option (State NativeGraph.Word) :=
  (candidateEvaluate (NativeGraph.algebra inst V) x y B p (NativeGraph.liftState s)).map
    (NativeGraph.projectState V B y p s)

/-- Both actual typed programs succeed with the same entire state. Their
physical result is obtained from those stored values at the selected owning
threads; no independent graph is executed in place of the source. -/
theorem complete (inst : NativeGraph.Instructions) (V : Nat) (x y : Tensor)
    (B p : Nat) (hB : 0 < B) (s : State NativeGraph.Word) :
    ∃ nodes t,
      Floating.evaluate (NativeGraph.algebra inst V) x y B p (NativeGraph.liftState s) = some nodes ∧
      candidateEvaluate (NativeGraph.algebra inst V) x y B p (NativeGraph.liftState s) = some nodes ∧
      t = NativeGraph.projectState V B y p s nodes ∧
      NativeGraph.evaluate inst V x y B p s = some t ∧ candidate inst V x y B p s = some t := by
  obtain ⟨nodes,run⟩ := Floating.terminates (NativeGraph.algebra inst V) x y B p hB (NativeGraph.liftState s)
  have candidateRun := (execution (NativeGraph.algebra inst V) x y B p (NativeGraph.liftState s)).symm.trans run
  exact ⟨nodes,_,run,candidateRun,rfl,by simp [NativeGraph.evaluate,run],by simp [candidate,candidateRun]⟩

#axiomsClean execution
#axiomsClean complete
end
end VeriTile.Bench.Repairs.TritonSoftmax.KernelNative
