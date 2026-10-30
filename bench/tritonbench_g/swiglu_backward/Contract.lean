import bench.tritonbench_g.swiglu_fwd.Contract
import bench.tritonbench_g.swiglu_backward.Proofs.Mathematical_WholeIndex

namespace VeriTile.Bench.Repairs.SwigluBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole

instance : Fintype Access where
  elems := {.x,.y,.dout,.dx,.dy,.out}
  complete a := by cases a <;> simp

instance (a : Access) (r : Bool) : Decidable (a.active r) :=
  inferInstanceAs (Decidable (a ≠ .out ∨ r = Bool.true))

/-- Actual data pointer and live enclosing span. Distinct Torch storage
objects are allowed to describe overlapping physical memory. -/
def matrixStart (byteBase : Nat → Nat) (m : Matrix) : Nat :=
  byteBase m.allocation + 4*m.base

def matrixEnd (byteBase : Nat → Nat) (rows width : Nat) (m : Matrix) : Nat :=
  matrixStart byteBase m + 4*((rows-1)*m.rowStride+width)

def SpanSeparate (byteBase : Nat → Nat) (rows widthA widthB : Nat) (a b : Matrix) : Prop :=
  matrixEnd byteBase rows widthA a ≤ matrixStart byteBase b ∨
  matrixEnd byteBase rows widthB b ≤ matrixStart byteBase a

instance (byteBase : Nat → Nat) (rows widthA widthB : Nat) (a b : Matrix) :
    Decidable (SpanSeparate byteBase rows widthA widthB a b) :=
  inferInstanceAs (Decidable (_ ∨ _))

structure LaunchContract (p : Prepared α call entry preexisting) (byteBase : Nat → Nat) : Prop where
  positive : 0 < p.ncols
  sameDevice : ∀ a : Access, a.active call.recompute → (a.view p).cudaDevice = call.input.cudaDevice
  rows32 : p.batchShape.prod < 2^31
  columns32 : p.ncols < 2^31
  stride32 : ∀ a : Access, a.active call.recompute → (a.view p).rowStride < 2^31
  padded32 : ∀ a : Access, a.active call.recompute → p.batchShape.prod*(a.view p).rowStride+p.ncols+1024 < 2^31
  gradientUnique : p.batchShape.prod ≤ 1 ∨ 2*p.ncols ≤ p.grad.matrix.rowStride
  outputUnique : call.recompute = Bool.true → p.batchShape.prod ≤ 1 ∨ p.ncols ≤ p.out.matrix.rowStride
  alignment : 0 < p.batchShape.prod → ∀ a : Access, a.active call.recompute →
    (byteBase (a.view p).allocation + 4*(a.view p).base) % 4 = 0
  gradientSeparate : 0 < p.batchShape.prod →
    SpanSeparate byteBase p.batchShape.prod (2*p.ncols) (2*p.ncols) p.grad.matrix p.xy.matrix ∧
    SpanSeparate byteBase p.batchShape.prod (2*p.ncols) p.ncols p.grad.matrix p.dout.matrix
  outputSeparate : 0 < p.batchShape.prod → call.recompute = Bool.true →
    SpanSeparate byteBase p.batchShape.prod p.ncols (2*p.ncols) p.out.matrix p.xy.matrix ∧
    SpanSeparate byteBase p.batchShape.prod p.ncols p.ncols p.out.matrix p.dout.matrix ∧
    SpanSeparate byteBase p.batchShape.prod p.ncols (2*p.ncols) p.out.matrix p.grad.matrix

def checkLaunch (p : Prepared α call entry preexisting) (byteBase : Nat → Nat) : Bool :=
  decide (0 < p.ncols) &&
  decide (∀ a : Access, a.active call.recompute → (a.view p).cudaDevice = call.input.cudaDevice) &&
  decide (p.batchShape.prod < 2^31) && decide (p.ncols < 2^31) &&
  decide (∀ a : Access, a.active call.recompute → (a.view p).rowStride < 2^31) &&
  decide (∀ a : Access, a.active call.recompute → p.batchShape.prod*(a.view p).rowStride+p.ncols+1024 < 2^31) &&
  decide (p.batchShape.prod ≤ 1 ∨ 2*p.ncols ≤ p.grad.matrix.rowStride) &&
  decide (call.recompute = Bool.true → p.batchShape.prod ≤ 1 ∨ p.ncols ≤ p.out.matrix.rowStride) &&
  decide (0 < p.batchShape.prod → ∀ a : Access, a.active call.recompute →
    (byteBase (a.view p).allocation+4*(a.view p).base)%4 = 0) &&
  decide (0 < p.batchShape.prod →
    SpanSeparate byteBase p.batchShape.prod (2*p.ncols) (2*p.ncols) p.grad.matrix p.xy.matrix ∧
    SpanSeparate byteBase p.batchShape.prod (2*p.ncols) p.ncols p.grad.matrix p.dout.matrix) &&
  decide (0 < p.batchShape.prod → call.recompute = Bool.true →
    SpanSeparate byteBase p.batchShape.prod p.ncols (2*p.ncols) p.out.matrix p.xy.matrix ∧
    SpanSeparate byteBase p.batchShape.prod p.ncols p.ncols p.out.matrix p.dout.matrix ∧
    SpanSeparate byteBase p.batchShape.prod p.ncols (2*p.ncols) p.out.matrix p.grad.matrix)

end VeriTile.Bench.Repairs.SwigluBackward
