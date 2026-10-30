import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.GegluTanhTriton
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- Shared selected-FP32 source. Both variants preserve the complete scalar
suffix, casts, constants, libdevice tanh and stores. Only the signed integer
tile-offset multiplication order changes. Python's constexpr 3*0.044715 is
folded to 0.134145 before scalar FP32 conversion. -/
def forwardKernel (a b c : RegionName) (count block : Nat) : ComputeKernel := triton {
  tl.region a = tl.float32, b = tl.float32, c = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  col_offsets = program_id * $(block) + tl.arange(0, $(block)).to(tl.int64)
  mask = col_offsets < $(count)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

def forwardIndexCommuted (a b c : RegionName) (count block : Nat) : ComputeKernel := triton {
  tl.region a = tl.float32, b = tl.float32, c = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  col_offsets = $(block) * program_id + tl.arange(0, $(block)).to(tl.int64)
  mask = col_offsets < $(count)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

def backwardKernel (dc a b : RegionName) (count block : Nat) : ComputeKernel := triton {
  tl.region dc = tl.float32, a = tl.float32, b = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  col_offsets = program_id * $(block) + tl.arange(0, $(block)).to(tl.int64)
  mask = col_offsets < $(count)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 0.134145 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}

def backwardIndexCommuted (dc a b : RegionName) (count block : Nat) : ComputeKernel := triton {
  tl.region dc = tl.float32, a = tl.float32, b = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  col_offsets = $(block) * program_id + tl.arange(0, $(block)).to(tl.int64)
  mask = col_offsets < $(count)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 0.134145 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}

inductive Channel where
  | c | da | db
  deriving DecidableEq, Repr

def outputRegion : Channel → RegionName
  | .c => "C" | .da => "A" | .db => "B"

def enabled (backward : Bool) (ch : Channel) : Prop :=
  if backward then ch ≠ .c else ch = .c

instance (b : Bool) (ch : Channel) : Decidable (enabled b ch) := by
  unfold enabled; infer_instance

def source (backward : Bool) (block count : Nat) : ComputeKernel :=
  if backward then backwardKernel "DC" "A" "B" count block
  else forwardKernel "A" "B" "C" count block

/-- The existing signed integer address-product commutation on the repaired
flat grid. No floating operation changes. -/
def indexCommutedSource (backward : Bool) (block count : Nat) : ComputeKernel :=
  if backward then backwardIndexCommuted "DC" "A" "B" count block
  else forwardIndexCommuted "A" "B" "C" count block

def selectedSource (backward candidate : Bool) (block count : Nat) : ComputeKernel :=
  if candidate then indexCommutedSource backward block count else source backward block count

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.GegluTanhTriton
