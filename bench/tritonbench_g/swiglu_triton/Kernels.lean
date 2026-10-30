import bench.tritonbench_g.swiglu_triton.Sources.Mathematical_Kernels
import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Separate FP32 host repair. Both JIT bodies, the silu helper, and all
calculate_settings branches are AST-identical to the original Python.
The current direct-storage entry adds checked contiguous FP32 metadata.
Historical padded-source declarations remain stable for downstream users.
The original source/proofs remain preserved, not retroactively certified. -/
namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile.Triton

abbrev forwardKernel (a b c : RegionName) (stride cols block : Nat) : ComputeKernel :=
  let a_ptr := a
  let b_ptr := b
  let c_ptr := c
  let n_cols := cols
  let BLOCK_SIZE := block
  triton {
  tl.ports inputs=[a_ptr, b_ptr], outputs=[c_ptr]
  tl.region a_ptr = tl.float32, b_ptr = tl.float32, c_ptr = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  c_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

def forwardIndexCommuted (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[a_ptr, b_ptr], outputs=[c_ptr]
  tl.region a_ptr = tl.float32, b_ptr = tl.float32, c_ptr = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += $(stride) * program_id
  b_ptr += $(stride) * program_id
  c_ptr += $(stride) * program_id
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

abbrev backwardKernel (dc a b : RegionName) (stride cols block : Nat) : ComputeKernel :=
  let dc_ptr := dc
  let a_ptr := a
  let b_ptr := b
  let n_cols := cols
  let BLOCK_SIZE := block
  triton {
  tl.ports inputs=[dc_ptr, a_ptr, b_ptr], outputs=[a_ptr, b_ptr]
  tl.region dc_ptr = tl.float32, a_ptr = tl.float32, b_ptr = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += program_id * $(stride)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
}

def backwardIndexCommuted (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  triton {
  tl.ports inputs=[dc_ptr, a_ptr, b_ptr], outputs=[a_ptr, b_ptr]
  tl.region dc_ptr = tl.float32, a_ptr = tl.float32, b_ptr = tl.float32
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += $(stride) * program_id
  a_ptr += $(stride) * program_id
  b_ptr += $(stride) * program_id
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
}

inductive Channel where
  | c | da | db
  deriving DecidableEq, Repr

def outputRegion : Channel → RegionName
  | .c => "C" | .da => "A" | .db => "B"

/-- Exactly the source configuration branches; HIP has 16 warps at the largest
blocks while CUDA has 32. Columns beyond 65536 are rejected before launch. -/
def warps (hip : Bool) (block : Nat) : Nat :=
  if block ≥ 32768 then (if hip then 16 else 32)
  else if block ≥ 8192 then 16 else if block ≥ 2048 then 8 else 4

def source (backward : Bool) (block cols : Nat) : ComputeKernel :=
  if backward then backwardKernel "DC" "A" "B" block cols block
  else forwardKernel "A" "B" "C" block cols block

/-- Actual contiguous rows have stride cols, including non-power-of-two widths. -/
def directSource (backward : Bool) (block cols : Nat) : ComputeKernel :=
  if backward then backwardKernel "DC" "A" "B" cols cols block
  else forwardKernel "A" "B" "C" cols cols block

/-- Existing typed-source candidate: exchange only the signed row-base
multiplication. Both actual stages retain their floating instruction order. -/
def directIndexCommutedSource (backward : Bool) (block cols : Nat) : ComputeKernel :=
  if backward then backwardIndexCommuted "DC" "A" "B" cols cols block
  else forwardIndexCommuted "A" "B" "C" cols cols block

def enabled (backward : Bool) (ch : Channel) : Prop :=
  if backward then ch ≠ .c else ch = .c

instance (b : Bool) (ch : Channel) : Decidable (enabled b ch) := by
  unfold enabled; infer_instance

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SwigluTriton
