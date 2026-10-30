import bench.tritonbench_g.fast_layernorm.Proofs.ReaderConnections
import bench.tritonbench_g.fast_layernorm.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FastLayernorm.Correctness
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
open scoped BigOperators
noncomputable section

/-- IO.lean fixes the canonical sources, masked logical channels, row/block
addresses and every output. These Real targets are independent formulas. -/
abbrev forwardIO := LogicalIO.forwardView KernelMath.algebra
abbrev backwardIO := LogicalIO.backwardView KernelMath.algebra
abbrev affineIO := LogicalIO.affineView KernelMath.algebra
/-- Independent row mean, inverse deviation and affine output. The sums use
logical width N; B is only the masked source tile. Output channels are Y/r/mu. -/
def forwardFormula (N B : Nat) (eps : ℝ) (bias : Bool) (v : LogicalIO.ForwardInputs B) :=
  let x := LogicalIO.entry (v.x)
  let mu := (∑ j ∈ Finset.range N, x j)/(N : ℝ)
  let r := 1/Real.sqrt ((∑ j ∈ Finset.range N, (x j-mu)^2)/(N : ℝ)+eps)
  LogicalIO.ForwardOutputs.mk (fun j => ((x j.val-mu)*r)*v.weight j +
    (if bias then v.bias j else 0)) r mu

/-- Saved mean/inverse deviation are logical inputs. No hidden forward run is
assumed. Total Real division also specifies the existing N=0 case. -/
def backwardFormula (N B : Nat) (v : LogicalIO.BackwardInputs B) :=
  let x := LogicalIO.entry (v.x)
  let q := fun j => LogicalIO.entry (v.upstream) j * LogicalIO.entry (v.weight) j
  let r := v.inverse
  let z := fun j => (x j-v.mean)*r
  fun (j : Fin B) =>
    (q j.val-(∑ k ∈ Finset.range N, q k)/(N : ℝ)-
      z j.val*(∑ k ∈ Finset.range N, q k*z k)/(N : ℝ))*r

/-- Both affine gradients sum over every row; the logical column test is
separate from the IO's independent DW/DB storage masks. -/
def affineFormula (rows N pid : Nat) (v : LogicalIO.AffineInputs rows) :=
  LogicalIO.AffineOutputs.mk
    (fun j => if pid*128+j.val<N then
      ∑ row : Fin rows, v.upstream row j *
        ((v.x row j-v.mean row)*v.inverse row) else 0)
    (fun j => if pid*128+j.val<N then ∑ row : Fin rows, v.upstream row j else 0)

/-- One forward program: normalized affine output, saved inverse deviation,
saved mean, actual source access safety and the complete exterior frame. -/
specification forward_kernel_correctness (N B pid : Nat) (covered : N ≤ B) (eps : ℝ) (bias : Bool) :
    Spec.Real (forwardIO N B pid eps bias ⊨ forwardFormula N B eps bias) :=
  ReaderConnections.forward N B pid covered eps bias (forwardFormula N B eps bias) rfl

/-- One backward program, including masked tails and the N=0 case. -/
specification backward_kernel_correctness (N B pid : Nat) (covered : N ≤ B) :
    Spec.Real (backwardIO N B pid ⊨ backwardFormula N B) :=
  (LogicalIO.backward_iff N B pid _).mpr (Interface.backward_real N B pid covered)

/-- All affine row-loop iterations and both independently masked outputs.
Counts above N retain the source's zero output for inactive input columns. -/
specification affine_kernel_correctness (rows N wc bc pid : Nat) (bias : Bool) :
    Spec.Real (affineIO rows N wc bc pid bias ⊨ affineFormula rows N pid) :=
  (LogicalIO.affine_iff rows N wc bc pid bias _).mpr (Interface.affine_real rows N wc bc pid bias)

#stmtSurfaceSubset forward_kernel_correctness ⊆ [LogicalIO.ForwardInputs,LogicalIO.ForwardOutputs,Spec.Real,LogicalProgramIO.Implements,forwardIO,forwardFormula]
#stmtSurfaceSubset backward_kernel_correctness ⊆ [LogicalIO.BackwardInputs,Spec.Real,LogicalProgramIO.Implements,backwardIO,backwardFormula]
#stmtSurfaceSubset affine_kernel_correctness ⊆ [LogicalIO.AffineInputs,LogicalIO.AffineOutputs,Spec.Real,LogicalProgramIO.Implements,affineIO,affineFormula]
#specNonCircular forwardFormula avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular backwardFormula avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#specNonCircular affineFormula avoiding [forwardKernel,backwardKernel,affineKernel,FP.Structural.exec]
#axiomsClean forward_kernel_correctness
#axiomsClean backward_kernel_correctness
#axiomsClean affine_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Correctness
