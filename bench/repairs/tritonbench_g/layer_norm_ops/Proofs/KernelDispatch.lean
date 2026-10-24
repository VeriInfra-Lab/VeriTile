import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Contract

namespace VeriTile.Bench.Repairs.LayerNormOps.KernelDispatch
open VeriTile.Triton FP.Structural Admission
noncomputable section

def epsilon (c : Common) : ℝ := match c.epsilon with
  | .float (some x) => x
  | _ => 0

def forward (c : ForwardCall) : ForwardParameters where
  x := "x"
  y := "y"
  weight := "weight"
  bias := "bias"
  residual := "residual"
  residualOut := "residualOut"
  mean := "mean"
  rstd := "rstd"
  xStride := c.common.columns
  yStride := c.common.columns
  residualStride := if c.residual.present then c.common.columns else 0
  residualOutStride := if c.residual.present then c.common.columns else 0
  columns := c.common.columns
  block := blockSize c.common.columns
  epsilon := epsilon c.common
  rms := c.common.rms.getD Bool.false
  hasResidual := c.residual.present
  storeResidual := c.residual.present
  hasBias := c.common.bias.present

def backward (c : BackwardCall) : BackwardParameters where
  x := "x"
  y := "y"
  weight := "weight"
  bias := "bias"
  dy := "dy"
  dx := "dx"
  dw := "dw"
  db := "db"
  dres := "dres"
  dresIn := "dresIn"
  mean := "mean"
  rstd := "rstd"
  xStride := c.common.columns
  yStride := if c.recompute.getD Bool.false then c.common.columns else 0
  dyStride := c.common.columns
  dxStride := c.common.columns
  dresStride := if c.dresidual.present then c.common.columns else 0
  dresInStride := 0
  meanStride := if c.common.rms.getD Bool.false then 0 else 1
  rstdStride := 1
  rows := c.common.rows
  columns := c.common.columns
  rowsPerProgram := c.rowWidth
  block := blockSize c.common.columns
  epsilon := epsilon c.common
  rms := c.common.rms.getD Bool.false
  hasDres := c.dresidual.present
  storeDres := Bool.false
  hasBias := c.common.bias.present
  recompute := c.recompute.getD Bool.false

theorem forward_admissible (c : ForwardCall) : KernelExecution.ForwardAdmissible (forward c) := by
  constructor <;> simp [forward]

theorem backward_admissible (c : BackwardCall) : KernelReadback.Admissible (backward c) := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro r hr
    simp only [KernelInvariant.reads,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with hr|hr|hr|hr|hr <;> subst r <;> simp [backward,Backward.RegionOutside]
  · constructor <;> simp [backward]
  · exact Nat.le_refl _
  · intro hy
    simp only [backward] at hy ⊢
    simp [hy]
  · simp [backward]

theorem checked_forward_block (c : ForwardCall) (h : forwardCheck c=Bool.true) :
    (forward c).columns≤(forward c).block :=
  (blockSize_properties _ ((forwardCheck_iff c).mp h).1.1 ((forwardCheck_iff c).mp h).1.2.1).1

theorem checked_backward_block (c : BackwardCall) (h : backwardCheck c=Bool.true) :
    (backward c).columns≤(backward c).block :=
  (blockSize_properties _ ((backwardCheck_iff c).mp h).1.1 ((backwardCheck_iff c).mp h).1.2.1).1

theorem epsilon_positive (c : Common) (h : CommonContract c) : 0<epsilon c := by
  obtain ⟨q,hq,hpos⟩ := h.2.2.2.2.2.2.1
  simpa [epsilon,hq] using hpos

theorem backward_empty_range (c : BackwardCall) (empty : c.common.rows=0)
    (A : Algebra α) [Inhabited α] (s : State α) :
    KernelReference.partials A (backward c) s=
      (fun _ => Forward.literal A 0,fun _ => Forward.literal A 0) := by
  simp [KernelReference.partials,Backward.stop,backward,empty,
    VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages.foldRange]

#axiomsClean forward_admissible
#axiomsClean backward_admissible
#axiomsClean checked_forward_block
#axiomsClean checked_backward_block
#axiomsClean backward_empty_range
end
end VeriTile.Bench.Repairs.LayerNormOps.KernelDispatch
