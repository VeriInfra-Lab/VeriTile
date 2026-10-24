import VeriTile.Triton
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivTriton
open VeriTile.Triton
set_option autoImplicit false

/-- Call-side pointer storage and signed64 scalar roles match the selected
Python annotations. Only block/reduction/log mode parameters are constexpr. -/
def forwardSignature : CallSignature where
  pointers := [("y_ptr",.fp32),("gt_ptr",.fp32),("loss_ptr",.fp32)]
  scalars := [("y_stride",.i64),("gt_stride",.i64),("loss_stride",.i64),("n_cols",.i64)]
  constexpr := ["BLOCK_SIZE","log_target","reduction"]
  readPorts := ["y_ptr","gt_ptr"]
  writePorts := ["loss_ptr"]

def backwardSignature : CallSignature where
  pointers := [("input_ptr",.fp32),("target_ptr",.fp32)]
  scalars := [("input_stride",.i64),("target_stride",.i64),("n_cols",.i64)]
  constexpr := ["BLOCK_SIZE","log_target"]
  readPorts := ["target_ptr"]
  writePorts := ["input_ptr"]

def forwardTemplate (y_ptr gt_ptr loss_ptr : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) : ComputeKernel :=
  let _REDUCTION_MODE_NONE : Nat := 0
  let BLOCK_SIZE : Nat := B
  let log_target : Bool := logTarget
  let y_stride_value : Nat := sy
  let gt_stride_value : Nat := st
  let loss_stride_value : Nat := so
  let n_cols_value : Int := N
  with_triton_call forwardSignature call_args
    [y_stride := y_stride_value, gt_stride := gt_stride_value, loss_stride := loss_stride_value, n_cols := n_cols_value] in
  triton {
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride
    base_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            loss = y_true * (tl.log(y_true) - y)
            loss = tl.where(y_true == 0.0, 0.0, loss)
        else:
            loss = tl.exp(y_true) * (y_true - y)
        loss = tl.where(mask, loss, 0.0)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.sum(loss, axis=0)
            tl.store(loss_ptr, loss)
            loss_ptr += 1
  }

def backwardTemplate (input_ptr target_ptr : RegionName) (so st N B : Nat)
    (logTarget : Bool) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  let log_target : Bool := logTarget
  let input_stride_value : Nat := so
  let target_stride_value : Nat := st
  let n_cols_value : Int := N
  with_triton_call backwardSignature call_args
    [input_stride := input_stride_value, target_stride := target_stride_value, n_cols := n_cols_value] in
  triton {
    pid = tl.program_id(0).to(tl.int64)
    input_ptr += pid * input_stride
    target_ptr += pid * target_stride
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)
        tl.store(input_ptr + offsets, res, mask=mask)
  }

def forwardCandidateTemplate (y_ptr gt_ptr loss_ptr : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) : ComputeKernel :=
  let _REDUCTION_MODE_NONE : Nat := 0
  let BLOCK_SIZE : Nat := B
  let log_target : Bool := logTarget
  let y_stride_value : Nat := sy
  let gt_stride_value : Nat := st
  let loss_stride_value : Nat := so
  let n_cols_value : Int := N
  with_triton_call forwardSignature call_args
    [y_stride := y_stride_value, gt_stride := gt_stride_value, loss_stride := loss_stride_value, n_cols := n_cols_value] in
  triton {
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += loss_stride * pid
    base_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            loss = y_true * (tl.log(y_true) - y)
            loss = tl.where(y_true == 0.0, 0.0, loss)
        else:
            loss = tl.exp(y_true) * (y_true - y)
        loss = tl.where(mask, loss, 0.0)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.sum(loss, axis=0)
            tl.store(loss_ptr, loss)
            loss_ptr += 1
  }

def backwardCandidateTemplate (input_ptr target_ptr : RegionName) (so st N B : Nat)
    (logTarget : Bool) : ComputeKernel :=
  let BLOCK_SIZE : Nat := B
  let log_target : Bool := logTarget
  let input_stride_value : Nat := so
  let target_stride_value : Nat := st
  let n_cols_value : Int := N
  with_triton_call backwardSignature call_args
    [input_stride := input_stride_value, target_stride := target_stride_value, n_cols := n_cols_value] in
  triton {
    pid = tl.program_id(0).to(tl.int64)
    input_ptr += input_stride * pid
    target_ptr += pid * target_stride
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)
        tl.store(input_ptr + offsets, res, mask=mask)
  }

def forwardSource (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) : ComputeKernel :=
  forwardTemplate "y_ptr" "gt_ptr" "loss_ptr" sy st so N B logTarget reduction

def backwardSource (so st N B : Nat) (logTarget : Bool) : ComputeKernel :=
  backwardTemplate "input_ptr" "target_ptr" so st N B logTarget

def forwardCandidateSource (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) : ComputeKernel :=
  forwardCandidateTemplate "y_ptr" "gt_ptr" "loss_ptr" sy st so N B logTarget reduction

def backwardCandidateSource (so st N B : Nat) (logTarget : Bool) : ComputeKernel :=
  backwardCandidateTemplate "input_ptr" "target_ptr" so st N B logTarget

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivTriton
