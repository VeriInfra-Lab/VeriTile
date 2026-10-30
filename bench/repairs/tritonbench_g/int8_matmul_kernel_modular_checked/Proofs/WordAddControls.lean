import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.WordAddBridge

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

/- ABI mismatches must fail before any fallback interpretation is available. -/

-- Rejected opcode.
/-- error: only the exact add.u32 instruction is supported -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.sat.s32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=True, pack=1,
  )
} : ComputeKernel)

-- Rejected register constraints.
/-- error: word-add ABI requires one output and two input 32-bit registers -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,0", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=True, pack=1,
  )
} : ComputeKernel)

-- Rejected argument order.
/-- error: word-add ABI requires dtype=tl.int32 and args=[accumulator, partial] -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[«partial», accumulator],
    dtype=tl.int32, is_pure=True, pack=1,
  )
} : ComputeKernel)

-- Rejected result width.
/-- error: word-add ABI requires dtype=tl.int32 and args=[accumulator, partial] -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int64, is_pure=True, pack=1,
  )
} : ComputeKernel)

-- Rejected purity.
/-- error: word-add ABI requires is_pure=True and pack=1 -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=False, pack=1,
  )
} : ComputeKernel)

-- Rejected packing.
/-- error: word-add ABI requires is_pure=True and pack=1 -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=True, pack=2,
  )
} : ComputeKernel)

-- Rejected keyword name.
/-- error: word-add ABI requires exactly constraints, args, dtype, is_pure, pack in source order -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", wrong="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=True, pack=1,
  )
} : ComputeKernel)

-- Rejected missing call.
/-- error: int8mod_triton requires exactly one source word-add call -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.zeros([1], dtype=tl.int32)
} : ComputeKernel)

-- Rejected duplicate call.
/-- error: int8mod_triton requires exactly one source word-add call -/
#guard_msgs in
#check (int8mod_triton {
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=True, pack=1,
  )
  accumulator = tl.inline_asm_elementwise(
    "add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, «partial»],
    dtype=tl.int32, is_pure=True, pack=1,
  )
} : ComputeKernel)

/-- Controls distinguish modular integer behavior from saturation and unbounded sums. -/
theorem add32_positive_overflow : signedAdd32 2147483647 1 = -2147483648 := by decide

theorem add32_negative_overflow : signedAdd32 (-2147483648) (-1) = 2147483647 := by decide

theorem add32_two_negative_words : signedAdd32 (-1) (-1) = -2 := by decide

#axiomsClean add32_positive_overflow
#axiomsClean add32_negative_overflow
#axiomsClean add32_two_negative_words
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
