import bench.repairs.tritonbench_g.triton_softmax.Proofs.Memory
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.FPKernel

namespace VeriTile.Bench.Repairs.TritonSoftmax.Floating
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
noncomputable section
set_option maxHeartbeats 4000000

abbrev access {α} [Inhabited α] := @SoftmaxTriton1.Floating.access α _
abbrev formula := @SoftmaxTriton1.Floating.formula
abbrev rowExpression := @SoftmaxTriton1.Floating.rowExpression
abbrev launchState := @SoftmaxTriton1.Floating.launchState

def evaluate {α} [Inhabited α] (M : Algebra α) (x y : Tensor) (B p : Nat) (s : State α) :=
  exec (SoftmaxTriton1.Floating.backend M) (softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
    y.rowStride y.colStride x.cols B) (launchState x p s)

/-- Only temporary register placement differs from the reused source. This
theorem expands both complete ASTs and proves equality of every memory cell,
including inactive cells and cells carrying non-real tags. -/
theorem source_memory {α} [Inhabited α] (M : Algebra α) (x y : Tensor)
    (B p : Nat) (hB : 0<B) (s t : State α) (runEq : evaluate M x y B p s=some t) :
    ∃ u,SoftmaxTriton1.Floating.evaluate M x y B p s=some u ∧ u.mem=t.mem := by
  simp [evaluate,exec,softmax_kernel,run,step,evalExpr,evalOp,store,
    State.setReg,numeric,natLt,bop,TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeKernel.surfaceBody,evalComputeOp,
    SoftmaxTriton1.Floating.backend,ComputeDType.eraseDType] at runEq
  subst t
  obtain ⟨u,hu⟩ := SoftmaxTriton1.Floating.source_terminates M x y B p hB s
  refine ⟨u,hu,?_⟩
  simp [SoftmaxTriton1.Floating.evaluate,exec,SoftmaxTriton1.softmax_kernel,run,step,
    evalExpr,evalOp,store,State.setReg,numeric,natLt,bop,TileShape.axisDim,hB,
    Option.bind,Function.comp_def,ComputeKernel.surfaceBody,evalComputeOp,
    SoftmaxTriton1.Floating.backend,ComputeDType.eraseDType] at hu
  subst u
  refine fold_observe State.mem _ _ _ _ ?_ ?_
  · rfl
  · intro a b i same
    split <;> simp_all [State.write]

theorem source {α} [Inhabited α] (M : Algebra α) :
    SourceContract access (evaluate M) (formula M) := by
  constructor
  intro x y B p s t values shape inj hB cover hp input run
  obtain ⟨u,hu,mem⟩ := source_memory M x y B p hB s t run
  have h := (SoftmaxTriton1.Floating.source M).row x y B p s u values
    shape inj hB cover hp input hu
  constructor
  · intro c hc
    simpa [access,SoftmaxTriton1.Floating.access,Access.read,mem] using h.1 c hc
  · intro r i off
    simpa [access,SoftmaxTriton1.Floating.access,mem] using h.2 r i off

theorem terminates {α} [Inhabited α] (M : Algebra α) (x y : Tensor)
    (B p : Nat) (hB : 0<B) (s : State α) : ∃ t,evaluate M x y B p s=some t := by
  simp [evaluate,exec,softmax_kernel,run,step,evalExpr,evalOp,store,
    State.setReg,numeric,natLt,bop,TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeKernel.surfaceBody,evalComputeOp,
    SoftmaxTriton1.Floating.backend,ComputeDType.eraseDType]

#axiomsClean source_memory
#axiomsClean source
#axiomsClean terminates
end
end VeriTile.Bench.Repairs.TritonSoftmax.Floating
