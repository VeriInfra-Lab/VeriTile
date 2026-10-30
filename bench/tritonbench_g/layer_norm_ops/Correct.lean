import bench.tritonbench_g.layer_norm_ops.Kernels
import bench.tritonbench_g.layer_norm_ops.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LayerNormOps.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

namespace Mathematics
def sum (n : Nat) (f : Nat → ℝ) := ∑ i ∈ Finset.range n, f i
def mean (n : Nat) (rms : Bool) (x : Nat → ℝ) := if rms then 0 else sum n x / n
def variance (n : Nat) (rms : Bool) (x : Nat → ℝ) :=
  sum n (fun i => (x i - mean n rms x)^2) / n
def rstd (n : Nat) (rms : Bool) (eps : ℝ) (x : Nat → ℝ) := 1 / Real.sqrt (variance n rms x + eps)
def normalized (n : Nat) (rms : Bool) (eps : ℝ) (x : Nat → ℝ) (i : Nat) :=
  (x i - mean n rms x) * rstd n rms eps x
def y (n : Nat) (rms bias : Bool) (eps : ℝ) (x w b : Nat → ℝ) (i : Nat) :=
  normalized n rms eps x i*w i + if bias then b i else 0

def dx (n : Nat) (rms extra : Bool) (x w dy dres : Nat → ℝ) (mu inv : ℝ) (i : Nat) :=
  let h := fun j => (x j - if rms then 0 else mu)*inv
  let g := fun j => w j*dy j
  (g i - h i*(sum n (fun j => h j*g j)/n) -
    (if rms then 0 else sum n g/n))*inv + (if extra then dres i else 0)

end Mathematics

def forwardFormula (p : ForwardParameters) (a : LogicalForwardIO.Inputs p) : LogicalForwardIO.Outputs p :=
  let x := fun j => LogicalForwardIO.extend a.x j + if p.hasResidual then LogicalForwardIO.extend a.residual j else 0
  ⟨fun i => Mathematics.y p.columns p.rms p.hasBias p.epsilon x
      (LogicalForwardIO.extend a.weight) (LogicalForwardIO.extend a.bias) i.val,
   fun i => x i.val,
   Mathematics.mean p.columns p.rms x,
   Mathematics.rstd p.columns p.rms p.epsilon x⟩

def backwardFormula (p : BackwardParameters) (pid : Nat) (a : LogicalBackwardIO.Inputs p) : LogicalBackwardIO.Outputs p :=
  let dx := fun row col => Mathematics.dx p.columns p.rms p.hasDres
    (LogicalBackwardIO.extend2 a.x row.val) (LogicalBackwardIO.extend a.weight)
    (LogicalBackwardIO.extend2 a.dy row.val) (LogicalBackwardIO.extend2 a.dres row.val)
    (LogicalBackwardIO.extend a.mean row.val) (LogicalBackwardIO.extend a.inverse row.val) col.val
  ⟨dx,
   fun row col => ((LogicalBackwardIO.extend2 a.x row.val col.val -
      if p.rms then 0 else LogicalBackwardIO.extend a.mean row.val)*LogicalBackwardIO.extend a.inverse row.val)*LogicalBackwardIO.extend a.weight col.val+
      if p.hasBias then LogicalBackwardIO.extend a.bias col.val else 0,
   dx,
   fun col => ∑ row ∈ Finset.Ico (Interface.rowStart p pid) (Interface.rowStop p pid),
      LogicalBackwardIO.extend2 a.dy row col.val * ((LogicalBackwardIO.extend2 a.x row col.val-
        if p.rms then 0 else LogicalBackwardIO.extend a.mean row)*LogicalBackwardIO.extend a.inverse row),
   fun col => ∑ row ∈ Finset.Ico (Interface.rowStart p pid) (Interface.rowStop p pid),
      LogicalBackwardIO.extend2 a.dy row col.val⟩

/-- Actual source access safety, successful execution, every active output and
complete exterior frame. The established region/stride alias conditions stay explicit. -/
specification layer_norm_ops_forward_kernel_real (p : ForwardParameters) (pid : Nat)
    (admissible : KernelExecution.ForwardAdmissible p) (covered : p.columns≤p.block) :
    Spec.Real (LogicalForwardIO.io Interface.realAlgebra p pid ⊨ forwardFormula p) :=
  (LogicalIO.forward_iff p pid).mpr (Interface.forward_real p pid admissible covered)

specification layer_norm_ops_backward_kernel_real (p : BackwardParameters) (pid : Nat)
    (admissible : KernelReadback.Admissible p) (covered : p.columns≤p.block) :
    Spec.Real (LogicalBackwardIO.io Interface.realAlgebra p pid ⊨ backwardFormula p pid) :=
  (LogicalIO.backward_iff p pid).mpr (Interface.backward_real p pid admissible covered)

#specNonCircular forwardFormula avoiding [forwardKernel,FP.Structural.exec,Forward.reference]
#specNonCircular backwardFormula avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#axiomsClean layer_norm_ops_forward_kernel_real
#axiomsClean layer_norm_ops_backward_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormOps.Correct
