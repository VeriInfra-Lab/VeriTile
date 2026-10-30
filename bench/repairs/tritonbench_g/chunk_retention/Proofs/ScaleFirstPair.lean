import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.PairIO
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ScaleFirstObservation
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ScaleFirstStageSafety
import bench.repairs.tritonbench_g.chunk_retention.Proofs.PublicCorrectness
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelPacket KernelStaging PublicIO
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- Only the forward-output stage changes; the three other complete original
kernels are preserved as the corresponding branches of the candidate packet. -/
theorem execution_transport (ops : Ops α) (literal : ℝ → α)
    (comm : ∀ x y, ops.mul x y=ops.mul y x) (a : Call) (s left : State (Term α))
    (ran : FP.Structural.exec (engine (withDot (algebra ops.zero) (nativeDot a))) a.kernel s=some left) :
    ∃ right, FP.Structural.exec (engine (withDot (algebra ops.zero) (nativeDot a))) (candidateKernel a) s=some right ∧
      (∀ (r : RegionName) off, mapCell (lower ops literal) (left.mem r off)=mapCell (lower ops literal) (right.mem r off)) ∧
      (∀ (r : RegionName) off, ¬Footprint a s r off → right.mem r off=left.mem r off) := by
  cases stage : a.stage with
  | forwardO=>
    have oldRan : FP.Structural.exec (engine (withDot (algebra ops.zero) dot))
        (ForwardOSource.kernel a.shape a.scale) s=some left := by
      simpa only [Call.kernel,nativeDot,stage] using ran
    obtain ⟨right,rightRan,cells,fr⟩:=ScaleFirstSource.comparison ops literal comm a.shape a.scale s left oldRan
    refine ⟨right,by simpa only [candidateKernel,nativeDot,stage] using rightRan,cells,?_⟩
    obtain ⟨t,ht,hm⟩:=ForwardOSource.source_memory (algebra ops.zero) dot a.shape a.scale s
    have eq : t=left := Option.some.inj (ht.symm.trans oldRan)
    subst t
    intro r off outside
    have exterior : r≠"o" ∨ ∀ i, ForwardOSource.active a.shape s i → off≠ForwardOSource.outputAddress a.shape s i := by
      by_cases hr : r="o"
      · right; intro i hi ho
        apply outside
        exact ⟨.output i,⟨stage,hi⟩,hr,ho⟩
      · exact Or.inl hr
    rw [fr r off exterior,hm]
    exact (ForwardOSource.frame (algebra ops.zero) dot a.shape a.scale s r off exterior).symm
  | forwardH|backwardH|backwardQKV=>
    refine ⟨left,?_,fun _ _=>rfl,fun _ _ _=>rfl⟩
    all_goals simpa only [candidateKernel,Call.kernel,stage] using ran

theorem equivalent (ops : Ops α) (literal : ℝ → α) (literals : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (comm : ∀ x y, ops.mul x y=ops.mul y x)
    (a : Call) (p grid : Nat → Nat) (legal : PublicIO.Legal a p grid) :
    Equivalent (original ops literal a p grid) (scaleFirst ops literal a p grid) := by
  intro xs bounds bounded initial binding
  obtain ⟨safe,left,leftRan,leftOut,leftFrame⟩:=
    native_implements ops literal literals castZero a p grid legal xs trivial bounds bounded initial binding
  let start := (sourceInvocation ops a p grid).initial initial
  have oldRan : FP.Structural.exec (engine (withDot (algebra ops.zero) (nativeDot a))) a.kernel start=some left := by
    simpa only [nativeIO,sourceInvocation,runSchedule_singleton,Invocation.run] using leftRan
  obtain ⟨right,rightRan,cells,exterior⟩:=execution_transport ops literal comm a start left oldRan
  refine ⟨safe,?_,left,right,leftRan,?_,?_,leftFrame,?_⟩
  · have h:=candidate_safe (algebra ops.zero) (nativeDot a) (Term.input "" 0) (lower ops literal)
      a start legal xs bounds bounded
    simpa only [scaleFirst,candidateInvocation,sourceInvocation,scheduleSafe_singleton,Invocation.Safe] using h
  · simpa only [scaleFirst,candidateInvocation,sourceInvocation,runSchedule_singleton,Invocation.run] using rightRan
  · intro o j active
    exact cells _ _
  · intro r off outside
    rw [exterior r off ((outside_iff (engine (withDot (algebra ops.zero) (nativeDot a)))
      (Term.input "" 0) (lower ops literal) a start xs r off).mp outside)]
    exact leftFrame r off outside

#axiomsClean execution_transport
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.PairIO
