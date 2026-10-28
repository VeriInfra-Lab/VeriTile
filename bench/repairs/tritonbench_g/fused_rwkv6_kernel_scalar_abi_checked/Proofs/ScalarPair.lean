import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.PairPrefix

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

theorem i32_original_prefix (c : Shape) (value : Int) :
    (kernelI32 c value).surfaceBody.take 14=(kernel c).surfaceBody.take 14 := rfl

theorem i32_candidate_prefix (c : Shape) (value : Int) :
    (earlyUKernelI32 c value).surfaceBody.take 14=(earlyUKernel c).surfaceBody.take 14 := rfl

theorem i32_suffix (c : Shape) (value : Int) :
    (earlyUKernelI32 c value).surfaceBody.drop 14=(kernelI32 c value).surfaceBody.drop 14 := rfl

/-- Complete original/candidate executions agree, including failure. -/
theorem i32_kernel_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (value : Int) (s : State α) :
    run M (earlyUKernelI32 c value).surfaceBody s=
      run M (kernelI32 c value).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelI32 c value).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelI32 c value).surfaceBody]
  rw [run_append,run_append,i32_original_prefix,i32_candidate_prefix,
    prefix_run,i32_suffix]

/-- Reordering the bias prefix preserves the complete access-safety judgment. -/
theorem i32_kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (value : Int) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (earlyUKernelI32 c value).surfaceBody s ↔
      Safety.traceSafeList M bounds (kernelI32 c value).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelI32 c value).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelI32 c value).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append,
    i32_original_prefix,i32_candidate_prefix,prefix_safe,prefix_run,i32_suffix]

#axiomsClean i32_kernel_run
#axiomsClean i32_kernel_safe

theorem i64_original_prefix (c : Shape) (value : Int) :
    (kernelI64 c value).surfaceBody.take 14=(kernel c).surfaceBody.take 14 := rfl

theorem i64_candidate_prefix (c : Shape) (value : Int) :
    (earlyUKernelI64 c value).surfaceBody.take 14=(earlyUKernel c).surfaceBody.take 14 := rfl

theorem i64_suffix (c : Shape) (value : Int) :
    (earlyUKernelI64 c value).surfaceBody.drop 14=(kernelI64 c value).surfaceBody.drop 14 := rfl

/-- Complete original/candidate executions agree, including failure. -/
theorem i64_kernel_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (value : Int) (s : State α) :
    run M (earlyUKernelI64 c value).surfaceBody s=
      run M (kernelI64 c value).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelI64 c value).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelI64 c value).surfaceBody]
  rw [run_append,run_append,i64_original_prefix,i64_candidate_prefix,
    prefix_run,i64_suffix]

/-- Reordering the bias prefix preserves the complete access-safety judgment. -/
theorem i64_kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (value : Int) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (earlyUKernelI64 c value).surfaceBody s ↔
      Safety.traceSafeList M bounds (kernelI64 c value).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelI64 c value).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelI64 c value).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append,
    i64_original_prefix,i64_candidate_prefix,prefix_safe,prefix_run,i64_suffix]

#axiomsClean i64_kernel_run
#axiomsClean i64_kernel_safe

theorem u64_original_prefix (c : Shape) (value : Nat) :
    (kernelU64 c value).surfaceBody.take 14=(kernel c).surfaceBody.take 14 := rfl

theorem u64_candidate_prefix (c : Shape) (value : Nat) :
    (earlyUKernelU64 c value).surfaceBody.take 14=(earlyUKernel c).surfaceBody.take 14 := rfl

theorem u64_suffix (c : Shape) (value : Nat) :
    (earlyUKernelU64 c value).surfaceBody.drop 14=(kernelU64 c value).surfaceBody.drop 14 := rfl

/-- Complete original/candidate executions agree, including failure. -/
theorem u64_kernel_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (value : Nat) (s : State α) :
    run M (earlyUKernelU64 c value).surfaceBody s=
      run M (kernelU64 c value).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelU64 c value).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelU64 c value).surfaceBody]
  rw [run_append,run_append,u64_original_prefix,u64_candidate_prefix,
    prefix_run,u64_suffix]

/-- Reordering the bias prefix preserves the complete access-safety judgment. -/
theorem u64_kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (value : Nat) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (earlyUKernelU64 c value).surfaceBody s ↔
      Safety.traceSafeList M bounds (kernelU64 c value).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelU64 c value).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelU64 c value).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append,
    u64_original_prefix,u64_candidate_prefix,prefix_safe,prefix_run,u64_suffix]

#axiomsClean u64_kernel_run
#axiomsClean u64_kernel_safe
theorem one_original_prefix (c : Shape) :
    (kernelOne c).surfaceBody.take 14=(kernel c).surfaceBody.take 14 := rfl

theorem one_candidate_prefix (c : Shape) :
    (earlyUKernelOne c).surfaceBody.take 14=(earlyUKernel c).surfaceBody.take 14 := rfl

theorem one_suffix (c : Shape) :
    (earlyUKernelOne c).surfaceBody.drop 14=(kernelOne c).surfaceBody.drop 14 := rfl

/-- Complete original/candidate executions agree, including failure. -/
theorem one_kernel_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (s : State α) :
    run M (earlyUKernelOne c).surfaceBody s=
      run M (kernelOne c).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelOne c).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelOne c).surfaceBody]
  rw [run_append,run_append,one_original_prefix,one_candidate_prefix,
    prefix_run,one_suffix]

/-- Reordering the bias prefix preserves the complete access-safety judgment. -/
theorem one_kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (earlyUKernelOne c).surfaceBody s ↔
      Safety.traceSafeList M bounds (kernelOne c).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernelOne c).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernelOne c).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append,
    one_original_prefix,one_candidate_prefix,prefix_safe,prefix_run,one_suffix]

#axiomsClean one_kernel_run
#axiomsClean one_kernel_safe


end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
