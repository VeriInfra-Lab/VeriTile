import bench.tritonbench_g.bmm_chunk_fwd.Proofs.KernelCandidate
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.CleanupSource
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeReferenceGrid

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelExecution.Raw
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open DirectFP NativeGraph
noncomputable section

/-- One source grid and the actual final cleanup DSL. Existing input/output
storage is supplied by the invocation; no wrapper or allocation is implemented. -/
def source (rest : Algebra Expr) (h : Host.Call) (chosen : Host.Config)
    (s : State Expr) : Option (State Expr) :=
  if h.shape.outputSize=0 then some s else do
    let last ← FPGrid.schedule (backend rest chosen) h chosen
      (FPGrid.allPrograms h chosen) s
    if h.causal then CleanupSource.schedule rest (.leaf .fp32 0) h Finset.univ.toList last
    else some last

def candidate (rest : Algebra Expr) (h : Host.Call) (chosen : Host.Config)
    (s : State Expr) : Option (State Expr) :=
  if h.shape.outputSize=0 then some s else do
    let last ← KernelCandidate.schedule (backend rest chosen) h chosen
      (FPGrid.allPrograms h chosen) s
    if h.causal then CleanupSource.schedule rest (.leaf .fp32 0) h Finset.univ.toList last
    else some last

theorem equivalent (rest : Algebra Expr) (h : Host.Call) (chosen : Host.Config) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) (s : State Expr) :
    source rest h chosen s=candidate rest h chosen s := by
  simp only [source,candidate,KernelCandidate.schedule_eq _ h _ ha hb]

/-- Source and candidate successfully execute the complete grid and cleanup,
with the same typed final state and the independently defined numerical result.
No successful-execution or numerical-correspondence law is a premise. -/
theorem complete (rest : Algebra Expr) (h : Host.Call) (chosen : Host.Config)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) (s : State Expr) (hm : Memory s) :
    ∃ t,source rest h chosen s=some t ∧ candidate rest h chosen s=some t ∧
      t.mem=NativeReference.kernelMemory h chosen s.mem ∧
      Memory t ∧ FPSuccess.RegionFrame h.Out s t := by
  by_cases empty : h.shape.outputSize=0
  · refine ⟨s,by simp [source,empty],by simp [candidate,empty],?_,hm,fun _ _ _ => rfl⟩
    simp [NativeReference.kernelMemory,empty]
  · obtain ⟨last,hl,href⟩ := NativeReference.grid_reference rest h chosen ha hb
      (FPGrid.allPrograms h chosen) s
    have hostRun : FPGrid.runHost (backend rest chosen) h chosen s=some last := by
      simpa only [FPGrid.runHost,show h.shape.outputSize≠0 from empty,↓reduceIte] using hl
    obtain ⟨last',hl',hmat⟩ := host_materialized rest h chosen s hm ha hb
    have same : last'=last := Option.some.inj (hl'.symm.trans hostRun)
    subst last'
    obtain ⟨last',hl',hframe,_,_⟩ := FPGrid.successful_host (backend rest chosen) h chosen s
    have same : last'=last := Option.some.inj (hl'.symm.trans hostRun)
    subst last'
    cases causal : h.causal
    · have run : source rest h chosen s=some last := by simp [source,empty,hl,causal]
      refine ⟨last,run,by rw [←equivalent rest h chosen ha hb]; exact run,?_,hmat,hframe⟩
      simpa only [NativeReference.kernelMemory,if_neg empty,NativeReference.clearMemory,
        show h.causal=Bool.false from causal,Bool.false_eq_true,↓reduceIte] using href
    · obtain ⟨t,ht,htm⟩ := CleanupSource.complete rest (.leaf .fp32 0) h last
      have bridge : t.mem=(clear h last).mem := by
        funext r off
        simpa only [clear,show h.causal=Bool.true from causal,↓reduceIte] using htm r off
      have run : source rest h chosen s=some t := by
        simpa only [source,if_neg empty,hl,Option.bind_some,causal,↓reduceIte] using ht
      refine ⟨t,run,by rw [←equivalent rest h chosen ha hb]; exact run,?_,?_,?_⟩
      · rw [bridge]
        simp only [clear,NativeReference.kernelMemory,NativeReference.clearMemory,
          if_neg empty,show h.causal=Bool.true from causal,↓reduceIte]
        rw [href]
      · simpa only [Memory,bridge] using clear_materialized h last hmat
      · intro r off hr
        rw [bridge,(clear_frame h last r off hr)]
        exact hframe r off hr

#axiomsClean equivalent
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelExecution.Raw
