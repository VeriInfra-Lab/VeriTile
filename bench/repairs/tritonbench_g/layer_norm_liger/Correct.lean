import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.LayerNormLiger.Correctness
open VeriTile.Triton FP.Structural
open LayerIO
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option autoImplicit false

namespace Mathematics
def sum (n : Nat) (f : Nat → ℝ) := ∑ i ∈ Finset.range n, f i
/-- Population statistics and affine normalization over logical columns only. -/
def mean (n : Nat) (x : Nat → ℝ) := sum n x / n
def variance (n : Nat) (x : Nat → ℝ) := sum n (fun i => (x i - mean n x)^2) / n
def rstd (n : Nat) (eps : ℝ) (x : Nat → ℝ) := (Real.sqrt (variance n x + eps))⁻¹
def y (n : Nat) (eps : ℝ) (x w b : Nat → ℝ) (i : Nat) :=
  (x i-mean n x)*rstd n eps x*w i+b i
/-- Backward target uses the forward's saved population statistics. -/
def xhat (x : Nat → ℝ) (mu inv : ℝ) (i : Nat) := (x i-mu)*inv
def dx (n : Nat) (x w dy : Nat → ℝ) (mu inv : ℝ) (i : Nat) :=
  (w i*dy i-(xhat x mu inv i*(sum n (fun j => xhat x mu inv j*(w j*dy j))/n)
    +sum n (fun j => w j*dy j)/n))*inv
def dw (rows : Nat) (x dy : Nat → Nat → ℝ) (mu inv : Nat → ℝ) (i : Nat) :=
  sum rows (fun row => dy row i*xhat (x row) (mu row) (inv row) i)
def db (rows : Nat) (dy : Nat → Nat → ℝ) (i : Nat) := sum rows (fun row => dy row i)

end Mathematics

def forwardFormula (p : Forward.Parameters) (a : LogicalForwardIO.Inputs p) : LogicalForwardIO.Outputs p := ⟨
  fun i => Mathematics.y p.columns p.epsilon (LogicalForwardIO.extend a.x)
    (LogicalForwardIO.extend a.weight) (LogicalForwardIO.extend a.bias) i.val,
  Mathematics.mean p.columns (LogicalForwardIO.extend a.x),
  Mathematics.rstd p.columns p.epsilon (LogicalForwardIO.extend a.x)⟩

def backwardFormula (p : Backward.Parameters) (pid : Nat) (a : LogicalBackwardIO.Inputs p) : LogicalBackwardIO.Outputs p := ⟨
  fun row col => Mathematics.dx p.columns (LogicalBackwardIO.extend2 a.x row.val)
    (LogicalBackwardIO.extend a.weight) (LogicalBackwardIO.extend2 a.dy row.val)
    (LogicalBackwardIO.extend a.mean row.val) (LogicalBackwardIO.extend a.inverse row.val) col.val,
  fun col => ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows),
    LogicalBackwardIO.extend2 a.dy row col.val *
      ((LogicalBackwardIO.extend2 a.x row col.val-LogicalBackwardIO.extend a.mean row)*LogicalBackwardIO.extend a.inverse row),
  fun col => ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows),
    LogicalBackwardIO.extend2 a.dy row col.val⟩

/-- Population normalization, affine Y and both saved statistics, with source
access safety and the complete exterior frame. Only the original mathematical
width and output-separation premises are required. -/
specification layer_norm_forward_kernel_real (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat)
    (hb : p.columns≤p.block) (hmy : p.mean≠p.y) (hmr : p.mean≠p.rstd) (hry : p.rstd≠p.y) :
    Spec.Real (LogicalForwardIO.io rest p pid ⊨ forwardFormula p) :=
  (LogicalIO.forward_iff rest p pid).mpr (LayerIO.forward_implements rest p pid hb hmy hmr hry)

/-- Every DX in the executed row interval and both partial DW/DB vectors.
The mathematical input/output separation and stride premises are unchanged;
empty groups still write their zero partial gradients. -/
specification layer_norm_backward_kernel_real (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat)
    (fresh : Logical.FreshDX p) (hwd : p.dw≠p.db) (hxd : p.dx≠p.dw) (hxb : p.dx≠p.db)
    (hb : p.columns≤p.block) (hs : p.columns≤p.dxStride) :
    Spec.Real (LogicalBackwardIO.io rest p pid ⊨ backwardFormula p pid) :=
  (LogicalIO.backward_iff rest p pid).mpr (LayerIO.backward_implements rest p pid fresh hwd hxd hxb hb hs)

#specNonCircular forwardFormula avoiding [Forward.kernel,FP.Structural.exec,Forward.reference]
#specNonCircular backwardFormula avoiding [Backward.kernel,FP.Structural.exec,Backward.reference]
#axiomsClean layer_norm_forward_kernel_real
#axiomsClean layer_norm_backward_kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Correctness
