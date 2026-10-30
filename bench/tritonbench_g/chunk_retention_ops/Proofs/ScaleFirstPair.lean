import bench.tritonbench_g.chunk_retention_ops.PairIO
import bench.tritonbench_g.chunk_retention_ops.Proofs.ScaleFirstSafety
import bench.tritonbench_g.chunk_retention_ops.Proofs.IOResults
import bench.tritonbench_g.chunk_retention.Proofs.ScaleFirstObservation
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PairIO
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelPacket KernelNative PublicIO
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
namespace Shared
export VeriTile.Bench.Repairs.ChunkRetention.Whole001 (Ops)
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelStaging (lower algebra dot)
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
  (kernel source_memory active outputAddress row column frame)
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource (comparison)
end Shared

/-- The sibling source identity is exact, including signed lane casts, padding,
FP32 dot options and all stores. Reusing its full proof does not impose PID or
floating-value bounds on this task. -/
theorem execution_transport (isa : ISA) (literal : ℝ → Word)
    (comm : ∀ x y, isa.mul x y=isa.mul y x) (a : Call) (s left : State Term)
    (ran : (sourceInvocation a).run s=some left) :
    ∃ right, (candidateInvocation a).run s=some right ∧
      (∀ (r : RegionName) off, mapCell (S.lower (sharedOps (wordOps isa)) literal) (left.mem r off)=
        mapCell (S.lower (sharedOps (wordOps isa)) literal) (right.mem r off)) ∧
      (∀ (r : RegionName) off, ¬Footprint a r off → right.mem r off=left.mem r off) := by
  cases stage : a.stage with
  | forwardOutput=>
    let start := (sourceInvocation a).initial s
    have oldRan : FP.Structural.exec
        (engine (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel.withDot (Shared.algebra (0 : Word)) Shared.dot))
        (Shared.kernel (KernelOutput.shape a.shape) a.scale) start=some left := by
      simpa only [start,sourceInvocation,Invocation.run,launch,Call.kernel,KernelNative.model,dotModel,stage,KernelOutput.source_identity] using ran
    obtain ⟨right,rightRan,cells,fr⟩:=Shared.comparison (sharedOps (wordOps isa)) literal comm
      (KernelOutput.shape a.shape) a.scale start left oldRan
    refine ⟨right,?_,cells,?_⟩
    · simpa only [start,candidateInvocation,sourceInvocation,Invocation.run,launch,candidateKernel,KernelNative.model,dotModel,stage,
        candidate_source_identity] using rightRan
    · obtain ⟨t,ht,hm⟩:=Shared.source_memory (Shared.algebra (0 : Word)) Shared.dot
        (KernelOutput.shape a.shape) a.scale start
      have eq : t=left := Option.some.inj (ht.symm.trans oldRan)
      subst t
      intro r off outside
      have exterior : r≠"o" ∨ ∀ i, Shared.active (KernelOutput.shape a.shape) start i →
          off≠Shared.outputAddress (KernelOutput.shape a.shape) start i := by
        by_cases hr : r="o"
        · right; intro i hi ho
          apply outside
          rw [Footprint,stage]
          change r="o" ∧ ∃ i, _ ∧ _
          refine ⟨hr,i,?_,?_⟩
          · simpa [Shared.active,Shared.row,Shared.column,KernelOutput.shape,start,sourceInvocation,
              launch,Invocation.initial,KernelReal.parameters,VeriTile.Bench.Repairs.ChunkLinearAttn.outputActive] using hi
          · simpa [Shared.outputAddress,Shared.row,Shared.column,KernelOutput.shape,start,sourceInvocation,
              launch,Invocation.initial,KernelReal.parameters,VeriTile.Bench.Repairs.ChunkLinearAttn.outputOffset] using ho
        · exact Or.inl hr
      rw [fr r off exterior,hm]
      exact (Shared.frame (Shared.algebra (0 : Word)) Shared.dot (KernelOutput.shape a.shape) a.scale start r off exterior).symm
  | forwardState|backwardState|backwardGradient=>
    refine ⟨left,?_,fun _ _=>rfl,fun _ _ _=>rfl⟩
    all_goals simpa only [candidateInvocation,sourceInvocation,Invocation.run,candidateKernel,Call.kernel,launch,stage] using ran

theorem equivalent (isa : ISA) (literal : ℝ → Word)
    (constants : S.LiteralLaws (sharedOps (wordOps isa)) literal)
    (comm : ∀ x y, isa.mul x y=isa.mul y x) (a : Call) :
    Equivalent (original isa literal a) (scaleFirst isa literal a) := by
  intro xs bounds bounded initial binding
  obtain ⟨safe,left,leftRan,leftOut,leftFrame⟩:=native_implements isa literal constants a xs trivial bounds bounded initial binding
  have oldRan : (sourceInvocation a).run initial=some left := by
    simpa only [nativeIO,sourceInvocation,runSchedule_singleton] using leftRan
  obtain ⟨right,rightRan,cells,exterior⟩:=execution_transport isa literal comm a initial left oldRan
  refine ⟨safe,?_,left,right,leftRan,?_,?_,leftFrame,?_⟩
  · have h:=candidate_safe model (dotModel a) (fun _ : Word=>default)
      (S.lower (sharedOps (wordOps isa)) literal) a xs bounds bounded initial
    simpa only [scaleFirst,candidateInvocation,sourceInvocation,scheduleSafe_singleton,Invocation.Safe] using h
  · simpa only [scaleFirst,runSchedule_singleton] using rightRan
  · intro o j active
    exact cells _ _
  · intro r off outside
    have notWritten : ¬Footprint a r off := by
      intro h
      obtain ⟨port,j,active,region,offset⟩:=footprint_covered a r off h
      exact (outside.1 port j active).elim (fun h=>h region) (fun h=>h offset)
    rw [exterior r off notWritten]
    exact leftFrame r off outside

theorem native_mul_comm (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (x y : Word) :
    (Numeric.nativeISA ports).mul x y=(Numeric.nativeISA ports).mul y x := by
  rw [Numeric.isa_eq u ports laws]
  change Numeric.mul32 x y=Numeric.mul32 y x
  unfold Numeric.mul32
  rw [FP.ScalarOpsLaws.mul_comm]

#axiomsClean execution_transport
#axiomsClean equivalent
#axiomsClean native_mul_comm
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.PairIO
