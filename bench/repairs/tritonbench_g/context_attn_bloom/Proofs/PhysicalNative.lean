import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.PhysicalConnection
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.RawInterface

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PhysicalKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open Bytes
noncomputable section

/-- Complete source grid, independent ordered native graph and fixed-profile
word/byte outputs. Premises are raw initial reads, address bounds and separation. -/
theorem correct_fp32 (g : Raw.Geometry) (r : Regions) (u : Inputs)
    (A : FlatAlloc) (B : Windows A.extent) (before : Memory)
    (observed : Observed (Raw.call g r u) B before)
    (valid : Host.Admissible g.arguments r A (layoutState B before))
    (labels : FPSuccess.Separate (Raw.call g r u))
    (separated : Separate (Raw.call g r u) B) (ops : Native.WordISA) :
    let c := Raw.call g r u
    let profile := KernelNative.profile g.arguments
    ∃ graph,FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)=some graph ∧
      graph.mem=(reference c B before).mem ∧
      (∀ i : Lane c,active c i →
        stores c B profile ops graph before (address c B i)=byte (referenceWord c B profile ops before i) i.2.2) ∧
      (∀ a,(¬∃ i : Lane c,active c i ∧ address c B i=a) → stores c B profile ops graph before a=before a) ∧
      (∀ rg∈Checked.readRegions r,∀ off,off < A.extent rg → ∀ j : Fin 4,
        stores c B profile ops graph before (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (∀ rg off,¬FPSuccess.writes c (order c) rg off → graph.mem rg off=(initial c B before).mem rg off) := by
  dsimp only
  let c := Raw.call g r u
  let profile := KernelNative.profile g.arguments
  obtain ⟨graph,run,referenceEq⟩ := FPReference.whole c Native.scalar Native.dots (order c) (initial c B before)
  obtain ⟨framed,frameRun,_,framing⟩ := FPSuccess.grid_success_frame c Native.scalar Native.dots
    labels (order c) (initial c B before) (Bytes.metadata c B before observed)
  have eq : framed=graph := Option.some.inj (frameRun.symm.trans run)
  subst framed
  refine ⟨graph,run,referenceEq,?_,frame c B profile ops graph before,?_,framing⟩
  · intro i hi
    rw [readback c B (layoutState B before) observed.symm valid profile ops graph before i hi]
    unfold outputWord referenceWord
    rw [referenceEq]
    rfl
  · intro rg read off bound j
    exact input_frame c A (layoutState B before) B observed.symm valid separated profile ops graph before rg read off bound j

#axiomsClean correct_fp32
#specNonCircular Bytes.referenceWord avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,kernel]
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PhysicalKernel
