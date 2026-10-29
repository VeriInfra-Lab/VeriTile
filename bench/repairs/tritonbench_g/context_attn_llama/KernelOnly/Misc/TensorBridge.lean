import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Correct
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Mathematics
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.FPValues
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Coverage

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 3000000

/-- Physical observation of executed Real output cells. Addresses are actual
Tensor byte addresses; this is a Real interpretation, not a floating encoding. -/
def realStores (r : Request) (final : BlockState) (before : Nat → ℝ) : Nat → ℝ :=
  writeback r (fun i => final.readMem (allocation r).flat
    ((allocation r).addr regions.out (r.call.offset i.1 i.2.1))) before

/-- The Real-erased shared source executes and gives the independent gathered,
finite-sentinel formula at every valid supplied output Tensor element. -/
theorem correct (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ) :
    ∃ final,
      Nonempty (Kernel.GridLaunchedOrdinary ((allocation r).flattenKernel
        (kernel r.call.layout regions .fp32 (Host.scale r.call.layout.D)).eraseDType.toAlgKernel)
        r.call.grid ((allocation r).flattenState (realInitial r heap read)) final) ∧
      (∀ b : Fin r.call.batches,∀ head : Fin r.call.layout.heads,∀ row,
        row<r.call.length b → ∀ dim : Fin r.call.layout.D,
        logical (realStores r final read) r.out (tensorIndex r.call b head row dim)=
          mathematical r heap read b head row dim) ∧
      (∀ a,(¬∃ i : Lane r,active r i ∧ address r i=a) → realStores r final read a=read a) ∧
      ReadFrame r read (realStores r final read) ∧
      (¬Needed r → realStores r final read=read) := by
  obtain ⟨result,run,values,_,_⟩ := Host.whole_call r.call (allocation r) (realInitial r heap read)
    (real_valid r heap h read) (allocation_disjoint r) (allocation_covered r) Bool.false
  refine ⟨result.2,run.2,?_,writeback_frame r _ read,writeback_inputs r heap h _ read,empty_writeback r _ read⟩
  intro b head row hr dim
  obtain ⟨idx,on,flat,offset⟩ := logical_coverage r heap h b head row hr dim
  have descriptor := h.checks.1.1 (views r).o (by simp)
  have shape : r.out.tensor.metadata.shape=[r.dimensions.queryRows,r.call.layout.heads,r.call.layout.D] := descriptor.1
  have contiguous : r.out.tensor.metadata.isContiguous=Bool.true := descriptor.2.2.1
  have fit := (flat_index r.call r.dimensions (extent r) h.checks.2.1 idx _ on).2
  have bound : tensorIndex r.call b head row dim<r.out.tensor.metadata.numel := by
    simpa [flat,Torch.TensorMetadata.numel,shape,Nat.mul_assoc] using fit
  rw [logical_direct _ r.out h.outAPI contiguous _ bound]
  have addressEq : address r (idx,r.call.lane row dim,⟨0,by decide⟩)=
      r.out.tensor.dataPtr.toNat+4*tensorIndex r.call b head row dim := by
    simp only [address,(flat_index r.call r.dimensions (extent r) h.checks.2.1 idx _ on).1,flat,Nat.add_zero]
  rw [←addressEq,realStores,writeback_readback r heap h _ read (idx,r.call.lane row dim,⟨0,by decide⟩) on]
  rw [offset]
  exact (values b head row hr (Host.full_coverage r.call b (complete_grid r heap h b b.isLt) row hr) dim).trans
    (expected_eq r heap read b head row dim)

#specNonCircular realStores avoiding [mathematical,mathematicalInputs,Call.logicalExpected]
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.TensorBridge

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 4000000

theorem fp_equiv_lanes (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    ∃ final,execute r heap=some final ∧
      (∀ idx : GridIndex r.call.grid,∀ i : TileIndex [r.call.layout.BM,r.call.layout.D],
        r.call.active idx i →
        (VeriTile.Torch.FP32.logicalValue (readBits (nativeStores r actual final heap))
          r.out.tensor h.outAPI.tensor (flatIndex r.call idx i)).bits=
          referenceWord r referenceOps heap (idx,i,⟨0,by decide⟩)) ∧
      (∀ a,(¬∃ i : Lane r,active r i ∧ address r i=a) → nativeStores r actual final heap a=heap a) ∧
      ReadFrame r heap (nativeStores r actual final heap) ∧
      (¬Needed r → nativeStores r actual final heap=heap) := by
  have eq := localLaws.operations_eq actual referenceOps
  subst referenceOps
  obtain ⟨final,run,bytes,frame,inputs,empty⟩ := native_bytes r heap h actual
  refine ⟨final,run,?_,frame,inputs,empty⟩
  intro idx i on
  exact logical_output r heap h actual _ idx i on (fun j => bytes (idx,i,j) on)

/-- Independent recurrence word at a logical packed-sequence output position,
evaluated at the physical owner of the stored row/channel replica. -/
def tensorReferenceWord (r : Request) (ops : NativeInstructions.ScalarOps) (heap : ByteMemory)
    (b head row dim : Nat) : Word :=
  let e := ((reference r heap).mem regions.out (r.call.output b head row dim)).read .real
  NativeInstructions.evaluate ops (NativeInstructions.plan r.call.layout.BM r.call.layout.D) e
    (NativeInstructions.owner r.call.layout.BM r.call.layout.D (row%r.call.layout.BM) dim)

/-- Public native result for every valid logical element of the supplied
Torch Tensor. Premises are ordinary inputs and pointwise primitive laws;
compiler realization and device/resource delivery stay external. -/
theorem fp_equiv (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    ∃ final,execute r heap=some final ∧
      (∀ b : Fin r.call.batches,∀ head : Fin r.call.layout.heads,∀ row,
        row<r.call.length b → ∀ dim : Fin r.call.layout.D,
        (VeriTile.Torch.FP32.logicalValue (readBits (nativeStores r actual final heap)) r.out.tensor
          h.outAPI.tensor (tensorIndex r.call b head row dim)).bits=
          tensorReferenceWord r referenceOps heap b head row dim) ∧
      (∀ a,(¬∃ i : Lane r,active r i ∧ address r i=a) → nativeStores r actual final heap a=heap a) ∧
      ReadFrame r heap (nativeStores r actual final heap) ∧
      (¬Needed r → nativeStores r actual final heap=heap) := by
  obtain ⟨final,run,values,frame,inputs,empty⟩ := fp_equiv_lanes r heap h actual referenceOps localLaws
  refine ⟨final,run,?_,frame,inputs,empty⟩
  intro b head row hr dim
  obtain ⟨idx,on,flat,offset⟩ := logical_coverage r heap h b head row hr dim
  have value := values idx (r.call.lane row dim) on
  rw [flat] at value
  have referenceEq : referenceWord r referenceOps heap (idx,r.call.lane row dim,⟨0,by decide⟩)=
      tensorReferenceWord r referenceOps heap b head row dim := by
    unfold referenceWord outputWord tensorReferenceWord
    rw [offset]
    rfl
  exact value.trans referenceEq

#specNonCircular tensorReferenceWord avoiding [FP.Structural.exec,FP.Structural.run,execute,nativeStores]
#axiomsClean fp_equiv
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.TensorBridge

