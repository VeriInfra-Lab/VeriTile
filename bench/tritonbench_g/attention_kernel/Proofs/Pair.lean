import bench.tritonbench_g.attention_kernel.PairIO
import bench.tritonbench_g.attention_kernel.Proofs.PairPrefix

namespace VeriTile.Bench.Repairs.AttentionKernel.Pair
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    run M (kernel p).surfaceBody s = run M (Source.kernel p).surfaceBody s := by
  calc
    _ = (run M ((kernel p).surfaceBody.take 18) s).bind
        (run M ((kernel p).surfaceBody.drop 18)) := by
      rw [←run_append,List.take_append_drop]
    _ = (run M ((Source.kernel p).surfaceBody.take 18) s).bind
        (run M ((Source.kernel p).surfaceBody.drop 18)) := by
      simp only [kernel,Source.kernel]
      rw [prefix_run,suffix_eq]
    _ = _ := by rw [←run_append,List.take_append_drop]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (kernel p).surfaceBody s ↔
      Safety.traceSafeList M bounds (Source.kernel p).surfaceBody s := by
  conv_lhs => rw [←List.take_append_drop 18 (kernel p).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 18 (Source.kernel p).surfaceBody]
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  simp only [kernel,Source.kernel]
  rw [prefix_safe,prefix_run,suffix_eq]

theorem invocation_run {β : Type} [Inhabited β] (M : Algebra β) (dot : Dot β)
    (p : Parameters) (pid : Program) (s : State β) :
    (job M dot p pid).run s = (IO.job M dot p pid).run s :=
  kernel_run (_root_.VeriTile.Triton.FP.ProfiledRewrite.engine (withDot M dot)) p _

theorem invocation_safe {β : Type} [Inhabited β] (M : Algebra β) (dot : Dot β)
    (p : Parameters) (pid : Program) (bounds : RegionName → Nat) (s : State β) :
    (job M dot p pid).Safe bounds s ↔ (IO.job M dot p pid).Safe bounds s :=
  kernel_safe (_root_.VeriTile.Triton.FP.ProfiledRewrite.engine (withDot M dot)) p bounds _

theorem mapped_run {β : Type} [Inhabited β] (M : Algebra β) (dot : Dot β)
    (p : Parameters) (work : List Program) (s : State β) :
    runSchedule (work.map (job M dot p)) s = runSchedule (work.map (IO.job M dot p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,runSchedule,invocation_run,ih]

theorem mapped_safe {β : Type} [Inhabited β] (M : Algebra β) (dot : Dot β)
    (p : Parameters) (work : List Program) (bounds : RegionName → Nat) (s : State β) :
    ScheduleSafe bounds (work.map (job M dot p)) s ↔
      ScheduleSafe bounds (work.map (IO.job M dot p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run,ih]

/-- Complete successful safe schedules on the entire original strided IO,
including empty grids, all caller grid permutations and both raw exterior frames. -/
theorem equivalent {α β : Type} [Inhabited β] (M : Algebra β) (dot : Dot β)
    (p : Parameters) (bases : Bases) (domain : Domain p bases) (hbm : 0<p.bm)
    (separate : Separated p) (work : List Program) (order : (programs (grid p)).Perm work)
    (encode : α → β) (observe : TileIndex (IO.TensorShape p) → β → α) :
    (IO.attentionIO M dot p work encode observe).Equivalent (variant M dot p work encode observe) := by
  intro vals _ bounds bounded initial binding
  have safe := IOGridSafety.safe M dot p bases domain hbm work
    (fun a ha => order.mem_iff.mpr ha) encode observe vals bounds bounded initial
  obtain ⟨result,ran,values,frame⟩ := RawKernel.complete M dot p hbm domain.launch separate initial work order
  have framed : (IO.attentionIO M dot p work encode observe).Frame vals initial result := by
    intro r off outside
    apply frame r off
    by_cases same : r=p.Out
    · right
      intro b hb h hh n hn d hd
      have miss := outside.1 (0 : Fin 1) (⟨b,hb⟩,⟨h,hh⟩,⟨n,hn⟩,⟨d,hd⟩,()) trivial
      simpa [IO.attentionIO,IO.outputs,IO.address,logicalAddress,same] using miss
    · exact Or.inl same
  refine ⟨safe,(mapped_safe M dot p work bounds initial).mpr safe,result,result,?_,?_,?_,framed,framed⟩
  · exact (IOGridSafety.schedule M dot p work initial).trans ran
  · exact (mapped_run M dot p work initial).trans ((IOGridSafety.schedule M dot p work initial).trans ran)
  · intro o j active
    rfl

#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent

end
end VeriTile.Bench.Repairs.AttentionKernel.Pair
