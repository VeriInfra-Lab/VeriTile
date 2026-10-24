import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.PairPrefix

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Pair
open VeriTile Triton FP.Structural Full
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (bf16 : Bool)
    (p : Parameters) (s : State α) :
    run M (candidateKernel bf16 p).surfaceBody s = run M (originalKernel bf16 p).surfaceBody s := by
  calc
    _ = (run M ((candidateKernel bf16 p).surfaceBody.take 18) s).bind
        (run M ((candidateKernel bf16 p).surfaceBody.drop 18)) := by
      rw [←run_append,List.take_append_drop]
    _ = (run M ((originalKernel bf16 p).surfaceBody.take 18) s).bind
        (run M ((originalKernel bf16 p).surfaceBody.drop 18)) := by rw [prefix_run,suffix_eq]
    _ = _ := by rw [←run_append,List.take_append_drop]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bf16 : Bool)
    (p : Parameters) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (candidateKernel bf16 p).surfaceBody s ↔
      Safety.traceSafeList M bounds (originalKernel bf16 p).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 18 (candidateKernel bf16 p).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 18 (originalKernel bf16 p).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append,prefix_safe,prefix_run,suffix_eq]

#axiomsClean kernel_run
#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Pair

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Pair
open VeriTile Triton FP.Structural Full Whole FPValue Interface
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

theorem invocation_run (p : Parameters) (pid : Program) (s : State Native.Expr) :
    (job p pid).run s = (Interface.job (Native.algebraFor p) (Native.modelFor p) p pid).run s :=
  _root_.VeriTile.Bench.Repairs.AttentionKernelAligned.Pair.kernel_run _ Bool.false p _

theorem invocation_safe (p : Parameters) (pid : Program) (bounds : RegionName → Nat)
    (s : State Native.Expr) :
    (job p pid).Safe bounds s ↔
      (Interface.job (Native.algebraFor p) (Native.modelFor p) p pid).Safe bounds s :=
  _root_.VeriTile.Bench.Repairs.AttentionKernelAligned.Pair.kernel_safe _ Bool.false p bounds _

theorem mapped_run (p : Parameters) (work : List Program) (s : State Native.Expr) :
    runSchedule (work.map (job p)) s =
      runSchedule (Interface.jobs (Native.algebraFor p) (Native.modelFor p) p work) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simpa only [Interface.jobs,List.map_cons,runSchedule,invocation_run] using
      congrArg (Option.bind ((Interface.job (Native.algebraFor p) (Native.modelFor p) p pid).run s))
        (funext ih)

theorem mapped_safe (p : Parameters) (work : List Program) (bounds : RegionName → Nat)
    (s : State Native.Expr) :
    ScheduleSafe bounds (work.map (job p)) s ↔
      ScheduleSafe bounds (Interface.jobs (Native.algebraFor p) (Native.modelFor p) p work) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [Interface.jobs,List.map_cons,ScheduleSafe,invocation_safe,invocation_run] at *
    simp only [ih]

/-- Original all-payload native input carrier and complete source schedule.
Both runs succeed safely, all output cells agree and all exterior cells frame. -/
theorem equivalent (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    (nativeIO p work).Equivalent (variant p work) := by
  intro vals _ bounds bounded initial binding
  have safe := native_safe p g work order vals bounds bounded initial
  obtain ⟨result,ran,values,frame⟩ := RawKernel.complete (Native.algebraFor p) (Native.modelFor p)
    p g separate initial work order
  have framed : (nativeIO p work).Frame vals initial result := by
    intro r off outside
    exact frame r off (outside_native p work vals r off outside)
  refine ⟨safe,(mapped_safe p work bounds initial).mpr safe,result,result,?_,?_,?_,framed,framed⟩
  · exact (jobs_run _ _ _ _ _).trans ran
  · exact (mapped_run p work initial).trans ((jobs_run _ _ _ _ _).trans ran)
  · intro o j active
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Pair

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Pair
open VeriTile Triton FP.Structural Full Whole FPValue Interface
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

theorem invocation_run (p : Parameters) (pid : Program) (s : State Native.Expr) :
    (job p pid).run s = (Interface.job (Native.algebraFor p) (Native.modelFor p) p pid).run s :=
  _root_.VeriTile.Bench.Repairs.AttentionKernelAligned.Pair.kernel_run _ Bool.true p _

theorem invocation_safe (p : Parameters) (pid : Program) (bounds : RegionName → Nat)
    (s : State Native.Expr) :
    (job p pid).Safe bounds s ↔
      (Interface.job (Native.algebraFor p) (Native.modelFor p) p pid).Safe bounds s :=
  _root_.VeriTile.Bench.Repairs.AttentionKernelAligned.Pair.kernel_safe _ Bool.true p bounds _

theorem mapped_run (p : Parameters) (work : List Program) (s : State Native.Expr) :
    runSchedule (work.map (job p)) s =
      runSchedule (Interface.jobs (Native.algebraFor p) (Native.modelFor p) p work) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simpa only [Interface.jobs,List.map_cons,runSchedule,invocation_run] using
      congrArg (Option.bind ((Interface.job (Native.algebraFor p) (Native.modelFor p) p pid).run s))
        (funext ih)

theorem mapped_safe (p : Parameters) (work : List Program) (bounds : RegionName → Nat)
    (s : State Native.Expr) :
    ScheduleSafe bounds (work.map (job p)) s ↔
      ScheduleSafe bounds (Interface.jobs (Native.algebraFor p) (Native.modelFor p) p work) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [Interface.jobs,List.map_cons,ScheduleSafe,invocation_safe,invocation_run] at *
    simp only [ih]

/-- Original all-payload native input carrier and complete source schedule.
Both runs succeed safely, all output cells agree and all exterior cells frame. -/
theorem equivalent (p : Parameters) (g : Guard p)
    (separate : Whole.Nonempty p → Separated p) (work : List Program)
    (order : (programs (grid p)).Perm work) :
    (nativeIO p work).Equivalent (variant p work) := by
  intro vals _ bounds bounded initial binding
  have safe := native_safe p g work order vals bounds bounded initial
  obtain ⟨result,ran,values,frame⟩ := RawKernel.complete (Native.algebraFor p) (Native.modelFor p)
    p g separate initial work order
  have framed : (nativeIO p work).Frame vals initial result := by
    intro r off outside
    exact frame r off (outside_native p work vals r off outside)
  refine ⟨safe,(mapped_safe p work bounds initial).mpr safe,result,result,?_,?_,?_,framed,framed⟩
  · exact (jobs_run _ _ _ _ _).trans ran
  · exact (mapped_run p work initial).trans ((jobs_run _ _ _ _ _).trans ran)
  · intro o j active
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Pair
