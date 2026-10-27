import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivOps
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

structure Parameters where
  (Y T O : RegionName)
  (rows cols block : Nat)
  deriving Repr

/-! Complete unchanged selected forward/backward bodies and the existing
output-row index-commutation candidates. Pointer types, runtime call bindings,
constexpr flags and block size remain outside each pasted Python body. -/
namespace KernelDSL

def forwardSignature : CallSignature where
  pointers := [("y_ptr",.fp32),("gt_ptr",.fp32),("loss_ptr",.fp32)]
  scalars := [("y_stride",.integer),("gt_stride",.integer),("loss_stride",.integer),
    ("n_cols",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE","log_target","reduction"]
  readPorts := ["y_ptr","gt_ptr"]
  writePorts := ["loss_ptr"]

def backwardSignature : CallSignature where
  pointers := [("target_ptr",.fp32),("new_grads_ptr",.fp32)]
  scalars := [("target_stride",.integer),("new_grads_stride",.integer),("n_cols",.integer)]
  constexpr := ["BLOCK_SIZE","log_target"]
  readPorts := ["target_ptr"]
  writePorts := ["new_grads_ptr"]

def forwardBody (p : Parameters) (log_target : Bool) (reduction : Nat) : ComputeKernel :=
  let y_ptr := p.Y
  let gt_ptr := p.T
  let loss_ptr := p.O
  let outputStride : Nat := if reduction = 0 then p.cols else 1
  let BLOCK_SIZE := p.block
  with_triton_call forwardSignature call_args [
    y_stride := p.cols, gt_stride := p.cols, loss_stride := outputStride, n_cols := p.cols] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    y_stride = tl.cast(y_stride, tl.int64)
    gt_stride = tl.cast(gt_stride, tl.int64)
    loss_stride = tl.cast(loss_stride, tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride

    base_offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == 0:  # _REDUCTION_MODE_NONE
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.where(mask, loss, 0.0)
            loss_sum += tl.sum(loss, axis=0)

    if reduction != 0:
        tl.store(loss_ptr, loss_sum)
  }

def forwardCandidateBody (p : Parameters) (log_target : Bool) (reduction : Nat) : ComputeKernel :=
  let y_ptr := p.Y
  let gt_ptr := p.T
  let loss_ptr := p.O
  let outputStride : Nat := if reduction = 0 then p.cols else 1
  let BLOCK_SIZE := p.block
  with_triton_call forwardSignature call_args [
    y_stride := p.cols, gt_stride := p.cols, loss_stride := outputStride, n_cols := p.cols] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    y_stride = tl.cast(y_stride, tl.int64)
    gt_stride = tl.cast(gt_stride, tl.int64)
    loss_stride = tl.cast(loss_stride, tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += loss_stride * pid

    base_offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == 0:  # _REDUCTION_MODE_NONE
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.where(mask, loss, 0.0)
            loss_sum += tl.sum(loss, axis=0)

    if reduction != 0:
        tl.store(loss_ptr, loss_sum)
  }

def backwardBody (p : Parameters) (log_target : Bool) : ComputeKernel :=
  let target_ptr := p.T
  let new_grads_ptr := p.O
  let BLOCK_SIZE := p.block
  with_triton_call backwardSignature call_args [
    target_stride := p.cols, new_grads_stride := p.cols, n_cols := p.cols] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    target_stride = tl.cast(target_stride, tl.int64)
    new_grads_stride = tl.cast(new_grads_stride, tl.int64)
    pid = tl.program_id(0).to(tl.int64)

    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride

    offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
        mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)

        tl.store(new_grads_ptr + offsets, res, mask=mask)
  }

def backwardCandidateBody (p : Parameters) (log_target : Bool) : ComputeKernel :=
  let target_ptr := p.T
  let new_grads_ptr := p.O
  let BLOCK_SIZE := p.block
  with_triton_call backwardSignature call_args [
    target_stride := p.cols, new_grads_stride := p.cols, n_cols := p.cols] in triton {
    n_cols = tl.cast(n_cols, tl.int64)
    target_stride = tl.cast(target_stride, tl.int64)
    new_grads_stride = tl.cast(new_grads_stride, tl.int64)
    pid = tl.program_id(0).to(tl.int64)

    target_ptr += pid * target_stride
    new_grads_ptr += new_grads_stride * pid

    offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
        mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)

        tl.store(new_grads_ptr + offsets, res, mask=mask)
  }

def forwardDefault (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardBody p Bool.false red

def forwardDefaultCommuted (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardCandidateBody p Bool.false red

def forwardLog (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardBody p Bool.true red

def forwardLogCommuted (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardCandidateBody p Bool.true red

def backwardDefault (p : Parameters) : ComputeKernel :=
  backwardBody p Bool.false

def backwardDefaultCommuted (p : Parameters) : ComputeKernel :=
  backwardCandidateBody p Bool.false

def backwardLog (p : Parameters) : ComputeKernel :=
  backwardBody p Bool.true

def backwardLogCommuted (p : Parameters) : ComputeKernel :=
  backwardCandidateBody p Bool.true

/- Each complete quotation declares its original ABI before pointer rebinding. -/
def forward (p : Parameters) (flag : Bool) (red : Nat) : ComputeKernel :=
  if flag then forwardLog p red else forwardDefault p red

def forwardCandidate (p : Parameters) (flag : Bool) (red : Nat) : ComputeKernel :=
  if flag then forwardLogCommuted p red else forwardDefaultCommuted p red

def backward (p : Parameters) (flag : Bool) : ComputeKernel :=
  if flag then backwardLog p else backwardDefault p

def backwardCandidate (p : Parameters) (flag : Bool) : ComputeKernel :=
  if flag then backwardLogCommuted p else backwardDefaultCommuted p

end KernelDSL

/-- Compatibility names used by the existing execution and mathematical proofs. -/
abbrev forward := KernelDSL.forward
abbrev backward := KernelDSL.backward

namespace KernelDSL
 theorem forward_eq (p : Parameters) (flag : Bool) (red : Nat) :
    forward p flag red = KldivOps.forward p flag red := rfl
 theorem backward_eq (p : Parameters) (flag : Bool) :
    backward p flag = KldivOps.backward p flag := rfl
end KernelDSL
end VeriTile.Bench.Repairs.KldivOps
