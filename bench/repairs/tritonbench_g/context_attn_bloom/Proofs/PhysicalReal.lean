import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PhysicalConnection
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.RawInterface
import bench.repairs.tritonbench_g.context_attn_bloom.Kernels

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PhysicalKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section

/-- Complete sequential Real source grid over raw geometry and flat memory:
independent normalized-block values and every exterior typed cell. -/
theorem correct_real (g : Raw.Geometry) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (valid : Host.Admissible g.arguments r A s) (disjoint : A.Disjoint)
    (covered : ∀ rg,rg∉A.regions → A.extent rg=0) (zero : s.undef=(fun _ _ => 0)) :
    Spec.Real (∃ final,
      RealStream.run g.arguments r (HostRun.allPrograms g.arguments) s=some final ∧
      RealStream.flatRun g.arguments r A (HostRun.allPrograms g.arguments) (A.flattenState s)=some (A.flattenState final) ∧
      (∀ idx i,Host.active g.arguments r s idx i →
        final.readMem r.out (RealStream.offset g.arguments r s idx i)=
          Host.result g.arguments r s idx i) ∧
      (∀ rg off,¬RealStream.writes g.arguments r s (HostRun.allPrograms g.arguments) rg off →
        final.mem rg off=s.mem rg off)) := by
  obtain ⟨final,source,flat,_,_,values,frame⟩ :=
    RealStream.complete g.arguments r A s valid disjoint covered zero (HostRun.allPrograms g.arguments)
  exact ⟨final,source,flat,fun idx i active => values idx (by simp [HostRun.allPrograms]) i active,frame⟩

/-- Both interpretations use the unchanged selected source quotation. -/
theorem shared_source (p : Layout) (r : Regions) (scale : ℝ) :
    kernel p r scale=VeriTile.Bench.Repairs.ContextAttnBloomSource.kernel
      r.query r.key r.value scale r.start r.sequence r.out r.tokens r.request r.prompt
      p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
      p.outRow p.outHead p.outDim p.requestRow p.requestToken p.kvGroup p.headDim p.BM p.D p.BN := rfl

#axiomsClean correct_real
#axiomsClean shared_source
#specNonCircular Host.result avoiding [RealStream.run,RealStream.flatRun,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PhysicalKernel
