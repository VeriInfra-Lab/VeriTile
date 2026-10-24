import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KldivTriton.LegacyPastedBody
open VeriTile.Triton
set_option autoImplicit false

/-- Full selected FP32 source. Host materialization supplies unit column
coordinates; each actual row stride remains an independent runtime argument.
Signed64 native arithmetic is connected separately to these mathematical indices. -/
def forwardTemplate (y_ptr gt_ptr loss_ptr : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) : ComputeKernel := triton {
  tl.region y_ptr = tl.float32, gt_ptr = tl.float32, loss_ptr = tl.float32
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(sy)
  gt_ptr += pid * $(st)
  loss_ptr += pid * $(so)
  base_offsets = tl.arange(0, $(B)).to(tl.int64)
  for i in range(0, $((N : Int)), $(B)) {
    offsets = i + base_offsets
    mask = offsets < $((N : Int))
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    if not logTarget {
      loss = y_true * (tl.log(y_true) - y)
      loss = tl.where(y_true == 0.0, 0.0, loss)
    } else {
      loss = tl.exp(y_true) * (y_true - y)
    }
    loss = tl.where(mask, loss, 0.0)
    if $(reduction) == $(0) {
      tl.store(loss_ptr + offsets, loss, mask=mask)
    } else {
      loss = tl.sum(loss, axis=0)
      tl.store(loss_ptr, loss)
      loss_ptr += $(1)
    }
  }
}

/-- Full backward source, including both retained pre-loop assignments. It
writes the unscaled derivative directly to the supplied input destination. -/
def backwardTemplate (input_ptr target_ptr : RegionName) (so st N B : Nat)
    (logTarget : Bool) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, target_ptr = tl.float32
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(so)
  target_ptr += pid * $(st)
  offsets = tl.arange(0, $(B)).to(tl.int64)
  mask = offsets < $((N : Int))
  for i in range(0, $((N : Int)), $(B)) {
    offsets = i + tl.arange(0, $(B)).to(tl.int64)
    mask = offsets < $((N : Int))
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    if not logTarget {
      res = target * -1
    } else {
      res = -tl.exp(target)
    }
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}

/-- Existing output-address candidate, applied to the repaired complete source.
Only the signed row-base product is commuted; every numerical node is retained. -/
def forwardCandidateTemplate (y_ptr gt_ptr loss_ptr : RegionName) (sy st so N B : Nat)
    (logTarget : Bool) (reduction : Nat) : ComputeKernel := triton {
  tl.region y_ptr = tl.float32, gt_ptr = tl.float32, loss_ptr = tl.float32
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(sy)
  gt_ptr += pid * $(st)
  loss_ptr += $(so) * pid
  base_offsets = tl.arange(0, $(B)).to(tl.int64)
  for i in range(0, $((N : Int)), $(B)) {
    offsets = i + base_offsets
    mask = offsets < $((N : Int))
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    if not logTarget {
      loss = y_true * (tl.log(y_true) - y)
      loss = tl.where(y_true == 0.0, 0.0, loss)
    } else {
      loss = tl.exp(y_true) * (y_true - y)
    }
    loss = tl.where(mask, loss, 0.0)
    if $(reduction) == $(0) {
      tl.store(loss_ptr + offsets, loss, mask=mask)
    } else {
      loss = tl.sum(loss, axis=0)
      tl.store(loss_ptr, loss)
      loss_ptr += $(1)
    }
  }
}

def backwardCandidateTemplate (input_ptr target_ptr : RegionName) (so st N B : Nat)
    (logTarget : Bool) : ComputeKernel := triton {
  tl.region input_ptr = tl.float32, target_ptr = tl.float32
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += $(so) * pid
  target_ptr += pid * $(st)
  offsets = tl.arange(0, $(B)).to(tl.int64)
  mask = offsets < $((N : Int))
  for i in range(0, $((N : Int)), $(B)) {
    offsets = i + tl.arange(0, $(B)).to(tl.int64)
    mask = offsets < $((N : Int))
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    if not logTarget {
      res = target * -1
    } else {
      res = -tl.exp(target)
    }
    tl.store(input_ptr + offsets, res, mask=mask)
  }
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
end VeriTile.Bench.Repairs.KldivTriton.LegacyPastedBody
