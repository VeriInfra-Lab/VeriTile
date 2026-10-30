import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.SafetyLoop

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.SourceSafety
open VeriTile.Triton FP.Structural Safety Execution Interface
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem safe_cons (j : Invocation α) (rest : List (Invocation α))
    (s t : State α) (bounds : RegionName → Nat) (hs : j.Safe bounds s)
    (he : j.run s=some t) (ht : ScheduleSafe bounds rest t) :
    ScheduleSafe bounds (j::rest) s := by
  refine ⟨hs,?_⟩
  intro u hu
  rw [he] at hu
  cases Option.some.inj hu
  exact ht

theorem native_jobs_safe (a : Arithmetic α) (g : GateInstructions α) (p : Parameters)
    (s : State α) (pid : Nat) (bytes : ByteValues s) (bounds : RegionName → Nat)
    (hb : Bounds p (GridExecution.atProgram s pid) bounds) :
    ScheduleSafe bounds (nativeJobs a g p pid) s := by
  let initial := GridExecution.atProgram s pid
  have bytes' : ByteValues initial := bytes
  let start := prefixExecution a.model p initial
  have hs := initialInvariant a Bool.true p initial
  obtain ⟨mid,hm,hi⟩ := loopExecution a Bool.true p initial start.val 0 (by omega) hs bytes'
  have hmid : FP.Structural.step a.model
      (.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)) start.val=some mid := by
    rw [countedExecution a p start.val _ hs.k]
    simpa [loopBody] using hm
  let norm := normExecution a p initial mid hi
  have hn : StoreContext p initial norm.val ∧
      norm.val.regs .real [16,16] "acc1"=some (accumulation a Bool.true p initial "W1" (steps p)) ∧
      norm.val.regs .real [16,16] "acc2"=some (accumulation a Bool.true p initial "W3" (steps p)) ∧
      norm.val.regs .real [16] "a_norm"=some (fun i => normalization a p initial i.1) :=
    normPost a p initial mid hi
  let gate := norm.val.setReg "accumulator" .real [16,16] (nativeValues a g p initial)
  have hg : gateExecution a g norm.val=some gate := by
    simp [gateExecution,hn.2.1,hn.2.2.1,hn.2.2.2,gate,nativeValues]
    rfl
  have hc : StoreContext p initial gate := by
    constructor <;> simp [gate,hn.1.memory,hn.1.m,hn.1.n,hn.1.pidm,hn.1.pidn,hn.1.outm,hn.1.outn]
  have hv : gate.regs .real [16,16] "accumulator"=some (nativeValues a g p initial) := by simp [gate]
  unfold nativeJobs
  refine safe_cons _ _ s start.val bounds ?_ ?_ ?_
  · exact prelude_safe a.model p initial bounds
  · rw [job_run]
    exact start.property
  refine safe_cons _ _ start.val mid bounds ?_ ?_ ?_
  · change traceSafeList a.model bounds
      [.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)] start.val
    simpa only [traceSafeList_unfold,hmid] using
      And.intro (counted_safe a Bool.true p initial start.val hs bytes' bounds hb) True.intro
  · rw [job_run]
    change run a.model [.forRangeInt "step" (.constInt 0) countOp (.constNat 1) (compiledBody p)] start.val=some mid
    simp only [run,hmid,Bind.bind]
    rfl
  refine safe_cons _ _ mid norm.val bounds ?_ ?_ ?_
  · exact norm_safe a p mid bounds
  · rw [job_run]; exact norm.property
  refine safe_cons _ _ norm.val gate bounds ?_ ?_ ?_
  · exact NativeStages.gate_safe a g bounds norm.val
  · rw [job_run]
    change run (NativeStages.gateBackend a g) NativeStages.gateCode norm.val=some gate
    rw [NativeStages.gate_exact,hg]
  apply (scheduleSafe_singleton _ _ _).mpr
  exact store_safe a p initial gate (nativeValues a g p initial) hc hv bounds hb


theorem jobs_safe (native : Bool) (a : Arithmetic α) (g : GateInstructions α) (p : Parameters)
    (s : State α) (pid : Nat) (bytes : ByteValues s) (bounds : RegionName → Nat)
    (hb : Bounds p (GridExecution.atProgram s pid) bounds) :
    ScheduleSafe bounds (jobs native a g p pid) s := by
  cases native
  · apply (scheduleSafe_singleton _ _ _).mpr
    exact source_safe a p (GridExecution.atProgram s pid) bytes bounds hb
  · exact native_jobs_safe a g p s pid bytes bounds hb

#axiomsClean native_jobs_safe
#axiomsClean jobs_safe
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.SourceSafety
