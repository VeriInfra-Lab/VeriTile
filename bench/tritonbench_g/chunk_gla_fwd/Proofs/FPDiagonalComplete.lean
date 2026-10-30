import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPDiagonalBranch
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem run_singleton (M : Algebra α) (st : ComputeStmt) (s : State α) :
    run M [st] s=step M st s := by
  rw [run]
  cases h : step M st s <;> simp only [Option.bind,run] <;> rfl

theorem source_memory (M : Algebra α) (split : Bool) (c : Layout) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine M) (kernel split c scale) s=some t ∧
      t.mem=(if enabled split c s then written M split c scale s else s).mem := by
  let start := entryExecution M split c scale s
  have hs : run (engine M) (parts split c scale).before (prologue M split c scale s).val=some start.val := by
    simpa only [run_append,(prologue M split c scale s).property,Option.bind_some] using start.property
  change ∃ t, run (engine M) (kernel split c scale).surfaceBody s=some t ∧ _
  rw [source_parts,run_append,(prologue M split c scale s).property]
  simp only [Option.bind_some]
  rw [run_singleton,step,condition_eval]
  by_cases hen : enabled split c s
  · simp only [hen,decide_true,↓reduceIte]
    obtain ⟨t,hm,hmem⟩ := loopExecution M split c scale s start.val
      (entryContext M split c scale s) (by cases split <;> simp [start,entryExecution])
    refine ⟨t,?_,hmem⟩
    have hloop : step (engine M) (.forRangeDyn "j" (parts split c scale).start (parts split c scale).stop
        (parts split c scale).stride (parts split c scale).body) start.val=some t := by
      have h0 : (parts split c scale).start = .constNat 0 := by cases split <;> rfl
      have h1 : (parts split c scale).stride = .constNat 1 := by cases split <;> rfl
      have hstop : FP.Structural.evalOp (engine M) none (parts split c scale).stop start.val=
          some (fun _ => columns split c s) := stop_eval M split c scale s
      rw [step,h0,h1,hstop]
      simp only [evalOp_unfold,Option.bind,Option.bind_some]
      exact hm
    rw [run_append,hs]
    simp only [Option.bind_some,run_singleton,hloop,↓reduceIte]
    rfl
  · simp only [hen,decide_false,Bool.false_eq_true,↓reduceIte,bind,Option.bind_some]
    exact ⟨(prologue M split c scale s).val,rfl,by cases split <;> simp [prologue]⟩

#specNonCircular written avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPDiagonal
