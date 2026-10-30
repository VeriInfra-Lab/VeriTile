import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.PairPrefix

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (s : State α) :
    run M (earlyUKernel c).surfaceBody s = run M (kernel c).surfaceBody s := by
  calc
    _ = (run M ((earlyUKernel c).surfaceBody.take 14) s).bind
        (run M ((earlyUKernel c).surfaceBody.drop 14)) := by
      rw [←run_append,List.take_append_drop]
    _ = (run M ((kernel c).surfaceBody.take 14) s).bind
        (run M ((kernel c).surfaceBody.drop 14)) := by rw [prefix_run,suffix_eq]
    _ = _ := by rw [←run_append,List.take_append_drop]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (earlyUKernel c).surfaceBody s ↔
      Safety.traceSafeList M bounds (kernel c).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 14 (earlyUKernel c).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 14 (kernel c).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append,prefix_safe,prefix_run,suffix_eq]

theorem invocation_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : α) (p : FusedRwkv6Kernel.Program) (s : State α) :
    (invocation M c scale p).run s = (Interface.invocation M c scale p).run s :=
  kernel_run (FP.ProfiledRewrite.engine M) c _

theorem invocation_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : α) (p : FusedRwkv6Kernel.Program)
    (bounds : RegionName → Nat) (s : State α) :
    (invocation M c scale p).Safe bounds s ↔
      (Interface.invocation M c scale p).Safe bounds s :=
  kernel_safe (FP.ProfiledRewrite.engine M) c bounds _

theorem mapped_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : α) (work : List FusedRwkv6Kernel.Program) (s : State α) :
    runSchedule (work.map (invocation M c scale)) s =
      runSchedule (work.map (Interface.invocation M c scale)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simpa only [List.map_cons,runSchedule,invocation_run] using
      congrArg (Option.bind ((Interface.invocation M c scale p).run s)) (funext ih)

theorem mapped_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : α) (work : List FusedRwkv6Kernel.Program)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (work.map (invocation M c scale)) s ↔
      ScheduleSafe bounds (work.map (Interface.invocation M c scale)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons p rest ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run] at *
    simp only [ih]

theorem schedule_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : α) (xs : Interface.Inputs α c) (s : State α) :
    runSchedule ((variant M c scale).schedule xs) s =
      runSchedule ((Interface.programIO M c scale).schedule xs) s := by
  by_cases live : Launches c
  · simpa only [variant,Interface.programIO,if_pos live] using mapped_run M c scale (programs c) s
  · simp only [variant,Interface.programIO,if_neg live]

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (scale : α) (xs : Interface.Inputs α c)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds ((variant M c scale).schedule xs) s ↔
      ScheduleSafe bounds ((Interface.programIO M c scale).schedule xs) s := by
  by_cases live : Launches c
  · simpa only [variant,Interface.programIO,if_pos live] using
      mapped_safe M c scale (programs c) bounds s
  · simp only [variant,Interface.programIO,if_neg live]

/-- Both complete grids run safely and successfully, with all typed outputs and
the exact exterior frame preserved. The original logical domain is unchanged. -/
theorem equivalent {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (hc : Sizes c) (plan : Plans c)
    (reduction : Source.ReductionMatches M c plan) (scale : α) :
    (Interface.programIO M c scale).Equivalent (variant M c scale) := by
  intro xs _ bounds bounded s binding
  obtain ⟨safe,result,ran,outputs,frame⟩ :=
    Interface.Proof.implements M c hc plan reduction scale xs trivial bounds bounded s binding
  refine ⟨safe,(schedule_safe M c scale xs bounds s).mpr safe,
    result,result,ran,?_,?_,frame,frame⟩
  · exact (schedule_run M c scale xs s).trans ran
  · intro o j active
    rfl

#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Pair
