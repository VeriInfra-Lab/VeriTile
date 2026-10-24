import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelReal
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelTensor

/-! Complete Tensor applications, physical observations and Python admission
correspondence, instantiated from the shared raw kernel execution proofs. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd.TensorBridge
noncomputable section

/-- Python admission and the public kernel precondition coincide. -/
theorem python_checks_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p :=
  checkArguments_iff p

/- Actual Tensor metadata, pointer, storage offset and logical input bindings;
four-byte output observations use the same complete blocked recurrence. -/
export KernelTensor (input_load initial_input output_address output_aligned output_bytes)

#axiomsClean python_checks_iff
#axiomsClean input_load
#axiomsClean initial_input
#axiomsClean output_address
#axiomsClean output_aligned
#axiomsClean output_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.TensorBridge

/- The Tensor results instantiate the same raw certificates as the primary
interfaces: HostContract.complete/HostGrid.flat_correct for Real arithmetic,
and NativeExecution.source_grid plus FPGridOutput.complete for native stores.
Direct.layout/nativeMetadata/nativeAllocation supply the explicit correspondence
from unchanged Tensor admission. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd.TensorBridge
open VeriTile.Triton
noncomputable section

/-- Complete kernel-grid correctness for the independent Doutᵀ @ A + residual
formula. Buffers are already bound; no wrapper or allocation is executed. -/
theorem bmm_chunk_bwd_real (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (inputs : Mathematical.Inputs) (choice : Fin 9) :
    ∃ final,KernelReal.run p a choice ((Direct.flat p a).flattenState (Direct.realState p inputs))=some final ∧
      (∀ b s g n,b<batch p → s<seqlen p → g<groups p → n<width p →
        final.readMem (Direct.flat p a).flat
          ((Direct.flat p a).addr (call p).Out (Direct.tensorIndex p b s g n))=
            Mathematical.product p inputs b s g n) ∧
      (∀ r off,¬Direct.OutputCell p a r off →
        final.mem r off=((Direct.flat p a).flattenState (Direct.realState p inputs)).mem r off) :=
  KernelReal.complete p h a inputs choice

#axiomsClean bmm_chunk_bwd_real
#specNonCircular Mathematical.product avoiding [KernelReal.run,Kernels.fp32Source]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.TensorBridge

namespace VeriTile.Bench.Repairs.BmmChunkBwd.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open DirectFP NativeGraph
noncomputable section

/-- Both complete readable kernels succeed with exactly the same source state.
That state is connected to the signed-INT64 native schedule and its actual
output stores under the documented SM90 compiler-staging model. No numerical
transformation assumptions or finite-word guards. -/
theorem bmm_chunk_bwd_equiv (rest : Algebra Expr) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (inputs : KernelTensor.Inputs p) (output : KernelTensor.Output p a)
    (choice : Fin 9) (heap : Heap) :
    ∃ final,KernelFP.run rest p a choice Bool.false (initial p a heap)=some final ∧
      KernelFP.run rest p a choice Bool.true (initial p a heap)=some final ∧
      ((p.tensor .a).numel≠0 →
        (KernelFP.native rest p h a choice (initial p a heap)).mem=
          (FPFlat.publish (Direct.placement p) (call p).Out (initial p a heap) final).mem) ∧
      Good (KernelFP.native rest p h a choice (initial p a heap)) ∧
      (∀ r off,¬(r=(call p).Out ∧ off<(p.tensor .a).numel) →
        (KernelFP.native rest p h a choice (initial p a heap)).mem r off=(initial p a heap).mem r off) ∧
      (∀ r (used : Used p r),r≠.output → ∀ i,i<(p.tensor r).numel →
        (initial p a heap).mem (KernelTensor.roleRegion r) i=.mk .fp32
          (.leaf (VeriTile.Torch.FP32.logicalValue (KernelTensor.readWord heap)
            (inputs.tensor r) (inputs.laws r used) i).bits)) ∧
      (∀ b s g n,b<batch p → s<seqlen p → g<groups p → n<width p →
        (output.tensor.storage.address (output.tensor.physicalIndex (Direct.tensorIndex p b s g n)))%4=0) ∧
      ((p.tensor .a).numel≠0 →
        FPGridOutput.Result (algebra rest) (dot choice) (call p) (tuning choice)
          (FPFlat.lift (call p) (Direct.placement p) (initial p a heap)) final) := by
  obtain ⟨final,source,candidate,stores,good,frame⟩ :=
    KernelFP.success rest p h a choice (initial p a heap) (initial_good p a heap)
  exact ⟨final,source,candidate,stores,good,frame,
    KernelTensor.initial_input p h a inputs heap,KernelTensor.output_aligned p h a output,
    fun nonempty => KernelFP.source_result rest p h a choice _ final nonempty source⟩

/- Independent blocked recurrence for every complete output coordinate. -/
export KernelFP (outputs candidate_setup_eq)

/-- Native output serialization preserves every byte outside the exact output
span, including all input buffers admitted by the disjoint-span contract. -/
theorem native_frame (rest : Algebra Expr) (arith : Arithmetic) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (choice : Fin 9) (heap : Heap) (address : Nat)
    (outside : ¬(a.pointer≤address ∧ address<a.pointer+4*(p.tensor .a).numel)) :
    outputHeap p a arith (KernelFP.native rest p h a choice (initial p a heap)) heap address=heap address :=
  output_frame p a arith _ heap address outside

#axiomsClean bmm_chunk_bwd_equiv
#axiomsClean native_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.TensorBridge
