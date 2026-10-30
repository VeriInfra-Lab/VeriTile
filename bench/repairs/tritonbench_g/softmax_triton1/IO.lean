import bench.repairs.tritonbench_g.softmax_triton1.Kernels
import VeriTile.Triton

/-! Canonical complete row IO, shared by the mathematical reader and retained
reciprocal application. Only the chosen source differs between variants. -/
namespace VeriTile.Bench.Repairs.SoftmaxTriton1.RealIO
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Independent row softmax, over exactly the original logical columns. -/
def reference (n : Nat) (xs : Nat → ℝ) (c : Nat) : ℝ :=
  Real.exp (xs c) / ∑ j ∈ Finset.range n, Real.exp (xs j)

/-- One actual row invocation, retaining storage offsets and complete active
read/write windows. The wrapper passes canonical contiguous strides. -/
def softmaxIO (out inp : RegionName) (ib ob n B : Nat) : MaskedKernelIO₁ where
  kernel := softmax_kernel out inp ib ob n 1 n 1 n B
  projection := by rfl
  inp := inp
  out := out
  B := B
  read := fun pid => ib + pid*n
  write := fun pid => ob + pid*n
  mask := fun _ i => i.val < n

/-- The retained reciprocal source has the same mathematical row result. -/
def reciprocalIO (out inp : RegionName) (ib ob n B : Nat) : MaskedKernelIO₁ :=
  { softmaxIO out inp ib ob n B with
    kernel := reciprocal_kernel out inp ib ob n 1 n 1 n B
    projection := by rfl }


end
end VeriTile.Bench.Repairs.SoftmaxTriton1.RealIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Logical
/-- Logical active-column access into the complete B-lane row. The underlying
input/output domains and all inactive entries remain unchanged. -/
def active {n B : Nat} (cover : n ≤ B) (xs : Fin B → ℝ) (j : Fin n) : ℝ :=
  xs (Fin.castLE cover j)
end VeriTile.Bench.Repairs.SoftmaxTriton1.Logical
