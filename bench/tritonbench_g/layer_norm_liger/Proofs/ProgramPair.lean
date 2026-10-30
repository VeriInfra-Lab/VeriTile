import bench.tritonbench_g.layer_norm_liger.Proofs.IOBindings
import bench.tritonbench_g.layer_norm_liger.Proofs.ForwardContract
import bench.tritonbench_g.layer_norm_liger.Proofs.CandidateSafety

namespace VeriTile.Bench.Repairs.LayerNormLiger.LayerIO.ProgramPair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

theorem forward_safe (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (pid : Nat) (v : ChannelValues α (fInputs p)) (bounds : RegionName → Nat)
    (hb : (forwardIO M R p pid).InBounds bounds v) (s : State α) :
    ScheduleSafe bounds ((forwardCandidate M R p pid).schedule v) s := by
  change ScheduleSafe bounds [fCandidateInvocation M R p pid] s
  rw [scheduleSafe_singleton]
  apply CandidateSafety.forward
  · intro i hi
    simpa [forwardIO,fMemory,fInputs,fCandidateInvocation] using hb.2.1 (0 : Fin 3) (i,PUnit.unit) hi
  · intro i hi
    exact hb.2.1 (1 : Fin 3) (i,PUnit.unit) hi
  · intro i hi
    exact hb.2.1 (2 : Fin 3) (i,PUnit.unit) hi
  · simpa [forwardIO,fMemory,fOutputs,fCandidateInvocation] using hb.2.2.1 (1 : Fin 3) PUnit.unit trivial
  · simpa [forwardIO,fMemory,fOutputs,fCandidateInvocation] using hb.2.2.1 (2 : Fin 3) PUnit.unit trivial
  · intro i hi
    simpa [forwardIO,fMemory,fOutputs,fCandidateInvocation] using hb.2.2.1 (0 : Fin 3) (i,PUnit.unit) hi

theorem backward_safe (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters)
    (pid : Nat) (v : ChannelValues α (bInputs p)) (bounds : RegionName → Nat)
    (hb : (backwardIO M R p pid).InBounds bounds v) (s : State α) :
    ScheduleSafe bounds ((backwardCandidate M R p pid).schedule v) s := by
  change ScheduleSafe bounds [bCandidateInvocation M R p pid] s
  rw [scheduleSafe_singleton]
  apply CandidateSafety.Backward.source_safe
  simpa only [bCandidateInvocation,b_pid] using backward_bounds M R p pid v bounds hb

/-- Common subexpression elimination preserves every typed Y/mean/rstd cell and
both exterior frames. All primitive payloads and source region aliases remain. -/
theorem forward (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters) (pid : Nat) :
    (forwardIO M R p pid).Equivalent (forwardCandidate M R p pid) := by
  intro vals _ bounds covered initial _
  let s := (fInvocation M R p pid).initial initial
  obtain ⟨left,ran,mem⟩ := Forward.complete_run M R p s
  obtain ⟨right,cran,cmem⟩ := Forward.Candidate.complete_run M R p s
  have framed : (forwardIO M R p pid).Frame vals initial left := by
    intro r o outside
    rw [mem]
    exact (Forward.reference_frame M R p s r o
      (forward_outside M R p pid vals initial r o outside)).trans
      (congrFun (congrFun (Invocation.initial_mem _ _) r) o)
  have same : right.mem=left.mem := cmem.trans mem.symm
  refine ⟨LayerIO.forward_safe M R p pid vals bounds covered initial,
    forward_safe M R p pid vals bounds covered initial,left,right,?_,?_,?_,framed,?_⟩
  · exact (runSchedule_singleton _ _).trans ran
  · exact (runSchedule_singleton _ _).trans cran
  · intro o i _
    change mapCell id (left.mem _ _) = mapCell id (right.mem _ _)
    rw [same]
  · intro r o outside
    rw [same]
    exact framed r o outside

/-- Hoisting the exact denominator conversion preserves the full row interval,
all DX cells, both partial vectors, empty loops, aliases and exterior frames. -/
theorem backward (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (pid : Nat) :
    (backwardIO M R p pid).Equivalent (backwardCandidate M R p pid) := by
  intro vals _ bounds covered initial _
  let s := (bInvocation M R p pid).initial initial
  obtain ⟨left,ran,mem⟩ := Backward.complete_run M R p s
  obtain ⟨right,cran,cmem⟩ := Backward.Candidate.complete_run M R p s
  have framed : (backwardIO M R p pid).Frame vals initial left := by
    intro r o outside
    rw [mem]
    exact (Backward.reference_frame M R p s r o
      (backward_outside M R p pid vals initial r o outside)).trans
      (congrFun (congrFun (Invocation.initial_mem _ _) r) o)
  have same : right.mem=left.mem := cmem.trans mem.symm
  refine ⟨LayerIO.backward_safe M R p pid vals bounds covered initial,
    backward_safe M R p pid vals bounds covered initial,left,right,?_,?_,?_,framed,?_⟩
  · exact (runSchedule_singleton _ _).trans ran
  · exact (runSchedule_singleton _ _).trans cran
  · intro o i _
    change mapCell id (left.mem _ _) = mapCell id (right.mem _ _)
    rw [same]
  · intro r o outside
    rw [same]
    exact framed r o outside

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.LayerIO.ProgramPair
