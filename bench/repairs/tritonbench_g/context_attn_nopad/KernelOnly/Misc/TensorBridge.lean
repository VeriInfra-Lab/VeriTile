import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Mathematics
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Coverage
import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.FPValues

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000

/-- Physical observation of executed Real output cells. Addresses are actual
Tensor byte addresses; this is a Real interpretation, not a floating encoding. -/
def realStores (r : Request) (final : BlockState) (before : Nat → ℝ) : Nat → ℝ :=
  writeback r (fun i => final.readMem (allocation r).flat
    ((allocation r).addr regions.out (FPReference.laneOffset r.call i.1 i.2.1))) before

/-- The Real-erased shared source executes successfully and gives independent
self-inclusive causal attention at every supplied output Tensor element.
The ordinary descriptor/metadata contract is the only premise. -/
theorem correct (r : Request) (heap : ByteMemory) (h : Contract r heap) (read : Nat → ℝ) :
    ∃ final,
      Nonempty (Kernel.GridLaunchedOrdinary ((allocation r).flattenKernel
        (kernel r.call regions).eraseDType.toAlgKernel) r.call.grid
        ((allocation r).flattenState (realInitial r heap read)) final) ∧
      (∀ b head t d,b<r.call.batches → head<r.call.q.heads → t<r.call.length b → d<r.call.q.dims →
        logical (realStores r final read) r.out (tensorIndex r.call (r.call.start b+t) head d)=
          mathematical r read b head t d) ∧
      (∀ a,(¬∃ i : Lane r,active r i ∧ address r i=a) → realStores r final read a=read a) ∧
      ReadFrame r read (realStores r final read) ∧
      (¬Needed r → realStores r final read=read) := by
  obtain ⟨result,run,values,_,_⟩ := Host.host_complete (real_memory r heap h read)
    (allocation_disjoint r) (allocation_covered r) (complete_grid r heap h) Bool.false
  refine ⟨result.1,run.1,?_,writeback_frame r _ read,writeback_inputs r heap h _ read,empty_writeback r _ read⟩
  intro b head t d hb hh ht hd
  obtain ⟨idx,i,on,flat,offset⟩ := logical_coverage r heap h b head t d hb hh ht hd
  have descriptor := h.checks.1.1 (views r).o (by simp)
  have shape : r.out.tensor.metadata.shape=[r.call.q.rows,r.call.q.heads,r.call.q.dims] := descriptor.1
  have contiguous : r.out.tensor.metadata.isContiguous=Bool.true := descriptor.2.2.1
  have capacity := (real_memory r heap h read).shape.qRows b hb
  have bound : tensorIndex r.call (r.call.start b+t) head d<r.out.tensor.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,shape,payloadCount,Nat.mul_assoc] using
      tensor_index_bound r heap h _ head d (by omega) hh hd
  rw [logical_direct _ r.out h.outAPI contiguous _ bound]
  have addressEq : address r (idx,i,⟨0,by decide⟩)=
      r.out.tensor.dataPtr.toNat+4*tensorIndex r.call (r.call.start b+t) head d := by
    simp only [address,(flat_index r.call regions (extent r) h.checks.2.1 idx i on).1,flat,
      tensorIndex,Nat.add_zero]
  rw [←addressEq,realStores,writeback_readback r heap h _ read (idx,i,⟨0,by decide⟩) on]
  rw [offset,values b head t d hb hh ht hd,expected_eq r heap h read b head t d hb hh ht hd]

#specNonCircular realStores avoiding [mathematical,mathematicalScore,Call.expected]
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.TensorBridge

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

/-- The selected SM80 instruction model executes the shared source and writes
the independent online recurrence into the supplied ordinary Torch Tensor.
Only pointwise scalar/fixed-size MMA instruction correspondences are assumed. -/
theorem fp_equiv_lanes (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    ∃ final,execute r heap=some final ∧
      (∀ idx : GridIndex r.call.grid,∀ i : TileIndex [r.call.layout.BM,r.call.layout.D],
        FPReference.laneActive r.call idx i →
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

/-- Independent recurrence word at a logical packed-sequence output position. -/
def tensorReferenceWord (r : Request) (ops : NativeInstructions.ScalarOps) (heap : ByteMemory)
    (b head t d : Nat) : Word :=
  let e := ((reference r heap).mem regions.out (r.call.output b head t d)).read .real
  NativeInstructions.evaluate ops (NativeProfiles.plan r.call.q.dims) e (NativeProfiles.ownerOf r.call.q.dims e)

/-- Public native result for every valid logical output element of the supplied
Torch Tensor, together with the full input-backing and output-complement frames. -/
theorem fp_equiv (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (actual referenceOps : NativeInstructions.ScalarOps)
    (localLaws : NativeInstructions.ScalarLaws actual referenceOps) :
    ∃ final,execute r heap=some final ∧
      (∀ b head t d,b<r.call.batches → head<r.call.q.heads → t<r.call.length b → d<r.call.q.dims →
        (VeriTile.Torch.FP32.logicalValue (readBits (nativeStores r actual final heap)) r.out.tensor
          h.outAPI.tensor (((r.call.start b+t)*r.call.q.heads+head)*r.call.q.dims+d)).bits=
          tensorReferenceWord r referenceOps heap b head t d) ∧
      (∀ a,(¬∃ i : Lane r,active r i ∧ address r i=a) → nativeStores r actual final heap a=heap a) ∧
      ReadFrame r heap (nativeStores r actual final heap) ∧
      (¬Needed r → nativeStores r actual final heap=heap) := by
  obtain ⟨final,run,values,frame,inputs,empty⟩ := fp_equiv_lanes r heap h actual referenceOps localLaws
  refine ⟨final,run,?_,frame,inputs,empty⟩
  intro b head t d hb hh ht hd
  obtain ⟨idx,i,on,flat,offset⟩ := logical_coverage r heap h b head t d hb hh ht hd
  have value := values idx i on
  simpa only [flat,referenceWord,outputWord,offset,tensorReferenceWord] using value

#specNonCircular tensorReferenceWord avoiding [FP.Structural.exec,FP.Structural.run,execute,nativeStores]
#axiomsClean fp_equiv
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.TensorBridge
