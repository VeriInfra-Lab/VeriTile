import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.RealOriginalDomain
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.KernelInterface

/-! Preserved raw-state and canonical physical-transport theorems. Public
readers expose shared IO judgments; these retain the complete historical
statements for consumers of the established execution foundation. -/

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section

/-- Complete finite grid over raw state and flat allocations, independent gathered
Real formula, exact output write frame and bounded input-region preservation. -/
theorem kernel_real (c : Host.Call) (allocation : FlatAlloc) (before : BlockState)
    (valid : Host.Valid c allocation before) (separate : allocation.Disjoint)
    (complete : ∀ rg,rg∉allocation.regions → allocation.extent rg=0) :
    ∃ final,Nonempty (Kernel.GridLaunchedOrdinary
      (allocation.flattenKernel (Program.kernel c.layout c.regions .fp32 Bool.false
        (Host.scale c.layout.D)).eraseDType.toAlgKernel)
      c.grid (allocation.flattenState before) final) ∧
      (∀ b : Fin c.batches,∀ h : Fin c.layout.heads,∀ row,row<c.length b →
        row<c.blocks*c.layout.BM → ∀ d : Fin c.layout.D,
        final.readMem allocation.flat (allocation.addr c.regions.out (c.output b h row d))=
          c.logicalExpected before b h row d) ∧
      BlockState.WriteWithin (c.writes allocation) (allocation.flattenState before) final ∧
      (∀ rg,rg∈c.inputs → ∀ off,off<allocation.extent rg →
        final.mem allocation.flat (allocation.addr rg off)=
          (allocation.flattenState before).mem allocation.flat (allocation.addr rg off)) := by
  obtain ⟨result,execution,output,frame,stable⟩ := Host.whole_call c allocation before valid separate complete Bool.false
  exact ⟨result.2,execution.2,output,frame,stable⟩

#specNonCircular Host.Call.logicalExpected avoiding [FP.Structural.exec,Kernel.GridLaunchedOrdinary]
#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section

/-- Full native source grid, equality to the independent ordered recurrence,
local-ISA observations and the exact exterior frame at raw kernel memory. -/
theorem kernel_native (c : Host.Call) (order : List (GridIndex c.grid))
    (before : State NativeInstructions.Expr) (metadata : FPSuccess.Metadata c before)
    (separate : c.regions.start≠c.regions.out ∧ c.regions.sequence≠c.regions.out ∧ c.regions.prompt≠c.regions.out)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    ∃ final,FPSuccess.runGrid c (NativeInstructions.backend c) order before=some final ∧
      final.mem=(FPReference.grid c (NativeInstructions.backend c) (NativeInstructions.support c) order before).mem ∧
      (∀ rg off consumer,NativeInstructions.evaluate actual (NativeInstructions.plan c.layout.BM c.layout.D)
        ((final.mem rg off).read .real) consumer=
        NativeInstructions.evaluate referenceOps (NativeInstructions.plan c.layout.BM c.layout.D)
          (((FPReference.grid c (NativeInstructions.backend c) (NativeInstructions.support c) order before).mem rg off).read .real) consumer) ∧
      (∀ rg off,¬FPSuccess.writes c order rg off → final.mem rg off=before.mem rg off) := by
  obtain ⟨final,run,memory⟩ := FPReference.whole c (NativeInstructions.backend c) (NativeInstructions.support c) order before
  refine ⟨final,run,memory,?_,?_⟩
  · intro rg off consumer
    rw [memory]
    exact NativeInstructions.evaluated_if_local actual referenceOps localLaws c _ consumer
  · intro rg off outside
    rw [memory]
    exact RawKernel.reference_frame c (NativeInstructions.backend c) (NativeInstructions.support c)
      order before metadata separate rg off outside

#axiomsClean kernel_native
#specNonCircular FPReference.grid avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid]
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
