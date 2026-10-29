import bench.examples.OnlineSoftmax.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-!
Proof-only mathematical counterparts. The authoritative native sources are
in ../Kernels.lean. Whole ComputeKernel erasure equalities below connect both
complete native programs to these mathematical proof helpers; they are not
the source of the active FP program pair.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
-/

namespace VeriTile.Bench.Examples.OnlineSoftmax.RealSource
open VeriTile Triton

/-- Stable softmax: subtract the max before exponentiating.
-/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  let code : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}
  ComputeKernel.fromKernelBody [xReg] [yReg] code.toAlgKernel.body

def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  stableSoftmaxKernel xReg yReg N

/-- Complete two-pass online softmax. The first pass computes m/l; the
second rereads the input row and stores every normalized output. -/
def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  let code : ComputeKernel := triton {
  pid := tl.program_id(0)
  m := tl.full([], -inf)
  l := tl.full([], 0)
  tl.for i in $(N) {
    xi := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l := libdevice.exp(m - m_new) * l + libdevice.exp(xi - m_new)
    m := m_new
  }
  pid := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x := tl.load($(xReg) + offs)
  e := libdevice.exp(x - m)
  y := e / l
  tl.store($(yReg) + offs, y)
}
  -- Both passes read the same input port.
  ComputeKernel.fromKernelBody [xReg] [yReg] code.toAlgKernel.body

/-- Proof-only mathematical prefix, connected to the full native source by
online_projection and the complete-body prefix decomposition. -/
def onlineNormalizerKernel (x y : RegionName) (N : Nat) : ComputeKernel :=
  .mk [x] [] ((onlineSoftmaxKernel x y N).surfaceBody.take 4)

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

theorem stable_projection (x y : RegionName) (B : Nat) :
    (VeriTile.Bench.Examples.OnlineSoftmax.Kernels.stableSoftmaxKernel x y B).eraseDType =
      stableSoftmaxKernel x y B := by
  simp [VeriTile.Bench.Examples.OnlineSoftmax.Kernels.stableSoftmaxKernel, stableSoftmaxKernel,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody, ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

#axiomsClean stable_projection

theorem online_projection (x y : RegionName) (N : Nat) :
    (VeriTile.Bench.Examples.OnlineSoftmax.Kernels.onlineSoftmaxKernel x y N).eraseDType =
      onlineSoftmaxKernel x y N := by
  simp [VeriTile.Bench.Examples.OnlineSoftmax.Kernels.onlineSoftmaxKernel, onlineSoftmaxKernel,
    ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.surfaceBody, ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

#axiomsClean online_projection

end VeriTile.Bench.Examples.OnlineSoftmax.RealSource
