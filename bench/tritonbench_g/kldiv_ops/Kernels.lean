import VeriTile.Triton.DSL.ScalarInt64While
import bench.tritonbench_g.kldiv_ops.Proofs.Sources.Parameters
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivOps
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

structure Parameters where
  (Y T O : RegionName)
  (rows cols block : Nat)
  deriving Repr

def forwardCallSignature : CallSignature where
  pointers := [("y_ptr",.fp32),("gt_ptr",.fp32),("loss_ptr",.fp32)]
  scalars := [("y_stride",.integer),("gt_stride",.integer),("loss_stride",.integer),("n_cols",.integer),("eps",.fp32)]
  constexpr := ["BLOCK_SIZE","log_target","reduction"]
  readPorts := ["y_ptr","gt_ptr"]
  writePorts := ["loss_ptr"]

def backwardCallSignature (target output : Storage) : CallSignature where
  pointers := [("target_ptr",match target with | .fp32 => .fp32 | .fp64 => .fp64 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5),
    ("new_grads_ptr",match output with | .fp32 => .fp32 | .fp64 => .fp64 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  scalars := [("target_stride",.integer),("new_grads_stride",.integer),("n_cols",.integer)]
  constexpr := ["BLOCK_SIZE","log_target"]
  readPorts := ["target_ptr"]
  writePorts := ["new_grads_ptr"]

def forwardKernel (y_ptr gt_ptr loss_ptr : RegionName) (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (log_target : Bool) (reduction : Nat) : ComputeKernel :=
  with_triton_call forwardCallSignature call_args [y_stride := y_stride, gt_stride := gt_stride,
    loss_stride := loss_stride, n_cols := n_cols] pointer_offsets [y_ptr := 0, gt_ptr := 0, loss_ptr := 0] in triton {
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

def forwardCandidateKernel (y_ptr gt_ptr loss_ptr : RegionName) (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (log_target : Bool) (reduction : Nat) : ComputeKernel :=
  with_triton_call forwardCallSignature call_args [y_stride := y_stride, gt_stride := gt_stride,
    loss_stride := loss_stride, n_cols := n_cols] pointer_offsets [y_ptr := 0, gt_ptr := 0, loss_ptr := 0] in triton {
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

def backwardKernel (target output : Storage) (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) (log_target : Bool) : ComputeKernel :=
  with_triton_call (backwardCallSignature target output) call_args [target_stride := target_stride,
    new_grads_stride := new_grads_stride, n_cols := n_cols]
    pointer_offsets [target_ptr := 0, new_grads_ptr := 0] in triton {
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

def backwardCandidateKernel (target output : Storage) (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) (log_target : Bool) : ComputeKernel :=
  with_triton_call (backwardCallSignature target output) call_args [target_stride := target_stride,
    new_grads_stride := new_grads_stride, n_cols := n_cols]
    pointer_offsets [target_ptr := 0, new_grads_ptr := 0] in triton {
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

/-- Storage variants instantiate the same copied source body. -/
def BackwardSource.defaultTarget (target output : Storage) (swapped : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  if swapped then backwardCandidateKernel target output T O st so N B Bool.false
  else backwardKernel target output T O st so N B Bool.false

def BackwardSource.logarithmic (target : MathStorage) (output : Storage) (swapped : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  if swapped then backwardCandidateKernel target.storage output T O st so N B Bool.true
  else backwardKernel target.storage output T O st so N B Bool.true

namespace KernelDSL
def forwardDefault (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardKernel p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1)
    p.cols p.block Bool.false red

def forwardLog (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardKernel p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1)
    p.cols p.block Bool.true red

def forwardDefaultCommuted (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardCandidateKernel p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1)
    p.cols p.block Bool.false red

def forwardLogCommuted (p : Parameters) (red : Nat) : ComputeKernel :=
  forwardCandidateKernel p.Y p.T p.O p.cols p.cols (if red=0 then p.cols else 1)
    p.cols p.block Bool.true red

def backwardDefault (p : Parameters) : ComputeKernel :=
  BackwardSource.defaultTarget .fp32 .fp32 Bool.false p.T p.O p.cols p.cols p.cols p.block

def backwardLog (p : Parameters) : ComputeKernel :=
  BackwardSource.logarithmic .fp32 .fp32 Bool.false p.T p.O p.cols p.cols p.cols p.block

def backwardDefaultCommuted (p : Parameters) : ComputeKernel :=
  BackwardSource.defaultTarget .fp32 .fp32 Bool.true p.T p.O p.cols p.cols p.cols p.block

def backwardLogCommuted (p : Parameters) : ComputeKernel :=
  BackwardSource.logarithmic .fp32 .fp32 Bool.true p.T p.O p.cols p.cols p.cols p.block

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
abbrev forward := KernelDSL.forward
abbrev backward := KernelDSL.backward
namespace KernelDSL
 theorem forward_eq (p : Parameters) (flag : Bool) (red : Nat) :
    forward p flag red = KldivOps.forward p flag red := rfl
 theorem backward_eq (p : Parameters) (flag : Bool) :
    backward p flag = KldivOps.backward p flag := rfl
end KernelDSL
end
end VeriTile.Bench.Repairs.KldivOps
