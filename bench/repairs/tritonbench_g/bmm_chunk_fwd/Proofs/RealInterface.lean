import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.RealSourceSafety
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.OutputGeometry
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelRealRaw

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
noncomputable section
open Classical

/-- The prior cleanup source execution supplies the concrete stage witness;
no second runner or output proof is introduced by the IO adapter. -/
def cleanupTrace (p : Parameters) (order : List (Cleanup.PID (parameters p)))
    (initial final : BlockState)
    (ran : CleanupReal.schedule (parameters p) order initial=some final) :
    AlgorithmExecution (order.map (fun pid=>.single (cleanupSource p) (cleanupLaunch p pid))) initial final := by
  induction order generalizing initial with
  | nil =>
    have same : initial=final := Option.some.inj ran
    subst final
    exact .nil _
  | cons pid rest ih =>
    cases head : exec (CleanupReal.kernel (parameters p)) (initial.withGridIndex pid) with
    | none => simp [CleanupReal.schedule,head] at ran
    | some middle =>
      have tail : CleanupReal.schedule (parameters p) rest middle=some final := by
        simpa [CleanupReal.schedule,head] using ran
      exact .cons ((AlgorithmStage.single_runs _ _ _ _).mpr head) (ih middle tail)

theorem cleanupTrace_safe (p : Parameters) (order : List (Cleanup.PID (parameters p)))
    (initial final : BlockState)
    (trace : AlgorithmExecution (order.map (fun pid=>.single (cleanupSource p) (cleanupLaunch p pid))) initial final)
    (bounds : RegionBounds) (output : ∀ off,off<p.shape.outputSize → off<bounds p.Out) :
    trace.Safe bounds := by
  induction order generalizing initial final with
  | nil => cases trace; trivial
  | cons pid rest ih =>
    cases trace with
    | cons head tail => exact ⟨cleanup_safe p pid bounds output _,ih _ _ tail⟩

theorem lift_runs (p : Parameters) (c : Host.Config) (v : RealInputs p) (initial final : BlockState)
    (nonempty : p.shape.outputSize≠0) (ran : KernelReal.Raw.Runs (parameters p) c initial final) :
    ∃ trace : AlgorithmExecution (stages p c) initial final,
      ∀ bounds,(realIO p c).InBounds bounds initial v → trace.Safe bounds := by
  change (if p.shape.outputSize=0 then _ else _) at ran
  rw [if_neg nonempty] at ran
  obtain ⟨middle,main,tail⟩ := ran
  have mainRuns : (AlgorithmStage.ordinary (mainSource p c) ((parameters p).grid c)).Runs initial middle := main
  cases causal : p.causal
  · have same : final=middle := by simpa [parameters,causal] using tail
    subst final
    have stage_eq : stages p c=[.ordinary (mainSource p c) ((parameters p).grid c)] := by
      simp [stages,nonempty,causal]
    rw [stage_eq]
    refine ⟨.cons mainRuns (.nil middle),?_⟩
    intro bounds bounded
    exact ⟨main_safe p c v initial nonempty bounds bounded,trivial⟩
  · have cleanup : CleanupReal.schedule (parameters p) Finset.univ.toList middle=some final := by
      simpa [parameters,causal] using tail
    let trace := cleanupTrace p Finset.univ.toList middle final cleanup
    have stage_eq : stages p c=
        .ordinary (mainSource p c) ((parameters p).grid c)::
          (Finset.univ.toList : List (Cleanup.PID (parameters p))).map
            (fun pid=>.single (cleanupSource p) (cleanupLaunch p pid)) := by
      simp [stages,nonempty,causal]
    rw [stage_eq]
    refine ⟨.cons mainRuns trace,?_⟩
    intro bounds bounded
    refine ⟨main_safe p c v initial nonempty bounds bounded,?_⟩
    exact cleanupTrace_safe p _ _ _ trace bounds (fun off ho=>bounded.2.1 (0 : Fin 1) ⟨off,ho⟩ trivial)

/-- Full mathematical kernel result through the audited shared IO relation.
All old raw execution/value/frame lemmas are reused, including ordinary-grid
alias behavior; the only added obligations connect declared source access
windows to their already proved trace bounds. -/
theorem real_implements (p : Parameters) (c : Host.Config) (hc : c∈Host.configurations) :
    (realIO p c).Implements (realFormula p) := by
  intro vals _ initial binding
  by_cases empty : p.shape.outputSize=0
  · have stage_eq : stages p c=[] := by simp [stages,empty]
    have vacuous : ∃ trace : AlgorithmExecution (stages p c) initial initial,
        ∀ bounds,trace.Safe bounds := by
      rw [stage_eq]
      exact ⟨.nil initial,fun _=>trivial⟩
    obtain ⟨trace,safe⟩ := vacuous
    refine ⟨initial,trace,fun bounds _=>safe bounds,?_,?_⟩
    · intro o i _
      have impossible : i.val<p.shape.outputSize := i.isLt
      omega
    · intro r off _ _
      rfl
  · obtain ⟨final,run,values,frame⟩ := KernelReal.Raw.complete (parameters p) c hc rfl rfl
      (logicalInputs p vals) initial (input_binding p c vals initial empty binding)
    obtain ⟨trace,safe⟩ := lift_runs p c vals initial final empty run
    refine ⟨final,trace,safe,?_,?_⟩
    · intro o i _
      obtain ⟨hb,hch,hg,hr,hn⟩ := output_coordinates p i.val i.isLt
      have value := values (outputBatch p i.val) (outputChunk p i.val) (outputGroup p i.val)
        (outputRow p i.val) (outputColumn p i.val) hb hch hg hr hn
      rw [output_offset] at value
      exact value
    · intro r off miss _
      apply frame r off
      rintro ⟨same,bound⟩
      rcases miss (0 : Fin 1) ⟨off,bound⟩ trivial with region|address
      · exact region same
      · exact address rfl

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
