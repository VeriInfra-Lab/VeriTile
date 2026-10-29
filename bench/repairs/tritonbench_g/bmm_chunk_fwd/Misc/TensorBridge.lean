import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelExecution
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Correct
import bench.repairs.tritonbench_g.bmm_chunk_fwd.FPEquiv
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelReal
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.CleanupOwnership
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.TensorBinding

/-! Complete Tensor applications, physical observations and Python admission
correspondence, instantiated from the shared raw kernel execution proofs. -/
namespace VeriTile.Bench.Repairs.BmmChunkFwd.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph DirectFP
noncomputable section
open Classical

theorem python_checks_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p :=
  checkArguments_iff p

/-- Ideal Real words at physical byte addresses. Only output word starts are
updated; this is a storage view of the actual final state, not a second result
formula. Native bytes are stated separately by FPEquiv. -/
def physicalWords (p : Metadata) (a : Direct.Allocation p) (final : BlockState)
    (before : Nat → ℝ) (address : Nat) : ℝ :=
  if ∃ i,i<(shape p).outputSize ∧ address=a.pointer+4*i then
    final.readMem (call p).Out ((address-a.pointer)/4)
  else before address

theorem physical_word (p : Metadata) (a : Direct.Allocation p) (final : BlockState)
    (before : Nat → ℝ) (i : Nat) (hi : i<(shape p).outputSize) :
    physicalWords p a final before (a.pointer+4*i)=final.readMem (call p).Out i := by
  unfold physicalWords
  rw [if_pos (show ∃ j,j<(shape p).outputSize ∧ a.pointer+4*i=a.pointer+4*j from ⟨i,hi,rfl⟩)]
  congr 1
  omega

theorem physical_frame (p : Metadata) (a : Direct.Allocation p) (final : BlockState)
    (before : Nat → ℝ) (address : Nat)
    (outside : ¬(a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize)) :
    physicalWords p a final before address=before address := by
  apply if_neg
  rintro ⟨i,hi,rfl⟩
  apply outside
  constructor <;> omega

/-- The same complete actual Real kernel invocation, now bound to original
Tensor payloads and every physical output word of the supplied Tensor. The
projection chooses an ideal Real carrier; it makes no TF32-to-Real claim. -/
theorem bmm_chunk_fwd_tensor_real (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (inputs : TensorBinding.Inputs p a)
    (output : TensorBinding.Output p a) (heap : DirectFP.Heap)
    (project : VeriTile.Torch.FP32.Word → ℝ)
    (chosen : Host.Config) (hc : chosen∈Host.configurations) :
    ∃ final,KernelReal.Runs p chosen
        (Direct.realState p (TensorBinding.idealInputs p a inputs heap project)) final ∧
      (∀ b ch g r n,b<(shape p).batch → ch<(shape p).chunks → g<(shape p).groups →
        r<(shape p).chunk → n<(shape p).chunk →
        physicalWords p a final (fun addr => project (TensorBinding.readWord heap addr))
          (output.tensor.storage.address (output.tensor.physicalIndex
            (Host.returnedOffset (call p) b ch g r n)))=
          Mathematical.product p (TensorBinding.idealInputs p a inputs heap project) b ch g r n) ∧
      (∀ r off,¬KernelReal.OutputCell p r off →
        final.mem r off=(Direct.realState p (TensorBinding.idealInputs p a inputs heap project)).mem r off) ∧
      (∀ address,¬(a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize) →
        physicalWords p a final (fun addr => project (TensorBinding.readWord heap addr)) address=
          project (TensorBinding.readWord heap address)) := by
  obtain ⟨final,runs,values,frame⟩ := KernelReal.complete p h
    (TensorBinding.idealInputs p a inputs heap project) chosen hc
  refine ⟨final,runs,?_,frame,physical_frame p a final _⟩
  intro b ch g r n hb hch hg hr hn
  have bound := KernelReal.offset_bound p b ch g r n hb hch hg hr hn
  rw [←TensorBinding.output_address p a output _ bound,physical_word p a final _ _ bound]
  exact values b ch g r n hb hch hg hr hn

/-- The values in source input memory are the original Tensor payloads at the
same queried physical pointers. INT64 IDs are decoded with their actual sign. -/
theorem bmm_chunk_fwd_input_tensors (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (inputs : TensorBinding.Inputs p a) (heap : Heap) :
    (∀ i,i<(p.tensor .a).numel →
      (⟨load32 heap (base p .a) i⟩ : VeriTile.Torch.FP32.Word)=
        VeriTile.Torch.FP32.logicalValue (TensorBinding.readWord heap) inputs.a inputs.aLaws i) ∧
    (∀ i,i<(p.tensor .b).numel →
      (⟨load32 heap (base p .b) i⟩ : VeriTile.Torch.FP32.Word)=
        VeriTile.Torch.FP32.logicalValue (TensorBinding.readWord heap) inputs.b inputs.bLaws i) ∧
    (p.hasSeq=Bool.true → ∀ i,i<(p.tensor .seq).numel →
      (load64 heap (base p .seq) i).toInt=
        (load64 heap inputs.seq.storageBase (inputs.seq.physicalIndex i)).toInt) :=
  ⟨TensorBinding.a_load p h a inputs heap,TensorBinding.b_load p h a inputs heap,
    TensorBinding.seq_load p h a inputs heap⟩

#axiomsClean python_checks_iff
#axiomsClean physical_word
#axiomsClean physical_frame
#axiomsClean bmm_chunk_fwd_tensor_real
#axiomsClean bmm_chunk_fwd_input_tensors
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.TensorBridge

/- Both Tensor applications and the primary raw interfaces reuse the exact
Host.frames_exist/main-value/cleanup certificates and the NativeReference
source-grid theorem. The unchanged Direct.real_binding and TensorBinding
lemmas instantiate the raw input pins and physical address correspondence. -/
namespace VeriTile.Bench.Repairs.BmmChunkFwd.TensorBridge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical

/-- Complete actual main and cleanup kernel invocation over arbitrary Real
inputs. Every legal configuration yields the independent full contraction;
every cell outside the output is unchanged. Allocation and host execution are
not premises of this kernel statement. -/
theorem bmm_chunk_fwd_real (p : Metadata) (h : InputContract p)
    (inputs : Mathematical.Inputs) (chosen : Host.Config) (hc : chosen∈Host.configurations) :
    ∃ final,KernelReal.Runs p chosen (Direct.realState p inputs) final ∧
      (∀ b ch g r n,b<(shape p).batch → ch<(shape p).chunks → g<(shape p).groups →
        r<(shape p).chunk → n<(shape p).chunk →
        final.readMem (call p).Out (Host.returnedOffset (call p) b ch g r n)=
          Mathematical.product p inputs b ch g r n) ∧
      (∀ r off,¬KernelReal.OutputCell p r off →
        final.mem r off=(Direct.realState p inputs).mem r off) :=
  KernelReal.complete p h inputs chosen hc

#axiomsClean bmm_chunk_fwd_real
#specNonCircular Mathematical.product avoiding [Kernels.source,KernelReal.Runs,CleanupReal.schedule]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.TensorBridge

namespace VeriTile.Bench.Repairs.BmmChunkFwd.TensorBridge
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph DirectFP
noncomputable section

/-- Structural floating result for arbitrary numerical machine primitives:
complete source execution, materialized observations, exact FP32 bytes, fresh
return identity and the entire exterior byte frame. -/
theorem bmm_chunk_fwd_native (rest : Algebra Expr) (actual : Arithmetic)
    (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (trials : List Host.Config) (chosen : Host.Config) (heap : Heap) :
    ∃ out, runWrapper rest actual p a trials chosen heap=.ok (some out) ∧
      out.object=a.object ∧ out.shape=(shape p).outputShape ∧
      out.device=(p.tensor .a).device ∧ out.dtype=.floating .fp32 ∧ out.pointer=a.pointer ∧
      (∀ r,Used p r → out.object≠a.inputObjects r) ∧
      (∀ off,off<(shape p).outputSize → Materialized (graph p (finalState rest p a trials chosen heap) off)) ∧
      (∀ off,off<(shape p).outputSize → ∀ b : Fin 4,
        out.heap (a.pointer+4*off+b.val)=
          byte (interpret actual (graph p (finalState rest p a trials chosen heap) off)) b.val) ∧
      (p.causal=Bool.true → ∀ off,Cleanup.Upper (call p) off → ∀ b : Fin 4,
        out.heap (a.pointer+4*off+b.val)=0) ∧
      (∀ address,¬(a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize) → out.heap address=heap address) := by
  let s := finalState rest p a trials chosen heap
  have runs := finalState_runs rest p a trials chosen heap
  refine ⟨result p a actual s heap,?_,rfl,rfl,rfl,rfl,rfl,a.fresh,?_,?_,?_,?_⟩
  · simp [runWrapper,(checkInputs_iff p).mpr h,runs.1,s]
  · intro off _
    exact runs.2.1 .fp32 (call p).Out off
  · intro off ho b
    exact output_byte p a actual s heap off ho b
  · intro causal off upper b
    have bound : off<(shape p).outputSize := by
      obtain ⟨pid,i,j,valid,eq⟩ := (Cleanup.member (call p) off).mp ((Cleanup.image (call p) off).mpr upper)
      rw [eq]
      exact Cleanup.offset_bound (call p) pid i j valid
    change outputHeap p a actual s heap _=0
    rw [output_byte p a actual s heap off bound b,
      execute_upper rest p trials chosen (initial p a heap) s runs.1 causal off upper]
    simp [interpret,byte]
  · exact output_frame p a actual s heap

/-- Every selected and tuning configuration receives a full native source
certificate from admission. These facts apply to every reached state, without
assuming successful execution or source-index safety. -/
theorem native_launches (rest : Algebra Expr) (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (trials : List Host.Config) (legal : ∀ c∈trials,c∈Host.configurations)
    (chosen : Host.Config) (hc : chosen∈Host.configurations) (nonempty : 0<(shape p).outputSize) :
    (∀ c∈chosen::trials, NativeProgram.Certificate (backend rest c) (call p) c (Direct.nativeLayout p h a)) ∧
      Cleanup.Native.addresses (call p)=Cleanup.addresses (call p) := by
  refine ⟨?_,Cleanup.Native.addresses_exact (call p)
    (Direct.nativeMetadata p h Cleanup.config Cleanup.config_member nonempty)⟩
  intro c member
  rcases List.mem_cons.mp member with rfl|member
  · exact Direct.nativeCertificate (backend rest c) p h a c hc nonempty
  · exact Direct.nativeCertificate (backend rest c) p h a c (legal c member) nonempty

/-- The only conditional step is transport across local numerical primitives at
actual graph operands. The reference can supply a concrete target-machine
literal, cast, binary and fixed K8 MMA coordinate implementation. -/
theorem bmm_chunk_fwd_if_local (rest : Algebra Expr) (actual reference : Arithmetic)
    (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (trials : List Host.Config) (chosen : Host.Config) (heap : Heap)
    (laws : TaskNumerics actual reference rest p a trials chosen heap) :
    ∃ out, runWrapper rest actual p a trials chosen heap=.ok (some out) ∧
      (∀ off,off<(shape p).outputSize → ∀ b : Fin 4,
        out.heap (a.pointer+4*off+b.val)=
          byte (interpret reference (graph p (finalState rest p a trials chosen heap) off)) b.val) ∧
      (∀ address,¬(a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize) → out.heap address=heap address) := by
  obtain ⟨out,run,_,_,_,_,_,_,_,values,_,frame⟩ := bmm_chunk_fwd_native rest actual p h a trials chosen heap
  refine ⟨out,run,?_,frame⟩
  intro off ho b
  rw [values off ho b,numerics_eq actual reference _ (laws off ho)]

/-- One complete source/candidate grid invocation, including the actual cleanup
kernel. Inputs and output are existing Tensor arguments. Every output byte at
its actual physical address equals the independent native recurrence. -/
theorem bmm_chunk_fwd_kernel (rest : Algebra Expr) (actual : Arithmetic)
    (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (inputs : TensorBinding.Inputs p a) (output : TensorBinding.Output p a)
    (chosen : Host.Config) (_chosenLegal : chosen∈Host.configurations) (heap : Heap) :
    ∃ final,KernelExecution.source rest p chosen (initial p a heap)=some final ∧
      KernelExecution.candidate rest p chosen (initial p a heap)=some final ∧
      Memory final ∧ FPSuccess.RegionFrame (call p).Out (initial p a heap) final ∧
      (∀ off,off<(shape p).outputSize → ∀ b : Fin 4,
        outputHeap p a actual final heap
          (output.tensor.storage.address (output.tensor.physicalIndex off)+b.val)=
          byte (interpret actual (NativeReference.resultGraph p chosen (initial p a heap).mem off)) b.val) ∧
      (∀ address,¬(a.pointer≤address ∧ address<a.pointer+4*(shape p).outputSize) →
        outputHeap p a actual final heap address=heap address) ∧
      (∀ i,i<(p.tensor .a).numel →
        (⟨load32 heap (base p .a) i⟩ : VeriTile.Torch.FP32.Word)=
          VeriTile.Torch.FP32.logicalValue (TensorBinding.readWord heap) inputs.a inputs.aLaws i) ∧
      (∀ i,i<(p.tensor .b).numel →
        (⟨load32 heap (base p .b) i⟩ : VeriTile.Torch.FP32.Word)=
          VeriTile.Torch.FP32.logicalValue (TensorBinding.readWord heap) inputs.b inputs.bLaws i) ∧
      (p.hasSeq=Bool.true → ∀ i,i<(p.tensor .seq).numel →
        (load64 heap (base p .seq) i).toInt=
          (load64 heap inputs.seq.storageBase (inputs.seq.physicalIndex i)).toInt) := by
  obtain ⟨final,run,candidate,hm,materialized,frame⟩ := KernelExecution.complete rest p chosen
    (initial p a heap) (initial_materialized p a heap)
  refine ⟨final,run,candidate,materialized,frame,?_,output_frame p a actual final heap,
    TensorBinding.a_load p h a inputs heap,TensorBinding.b_load p h a inputs heap,
    TensorBinding.seq_load p h a inputs heap⟩
  intro off bound b
  rw [←TensorBinding.output_address p a output off bound,output_byte p a actual final heap off bound b]
  simp only [graph,NativeReference.resultGraph,hm]

/-- Source/candidate equality includes cleanup registers and every typed memory
cell, rather than merely an equality of readback functions. -/
theorem bmm_chunk_fwd_execution (rest : Algebra Expr) (p : Metadata)
    (chosen : Host.Config) (s : State Expr) (hm : Memory s) :
    ∃ final,KernelExecution.source rest p chosen s=some final ∧
      KernelExecution.candidate rest p chosen s=some final ∧
      final.mem=NativeReference.completeMemory p chosen s.mem ∧
      Memory final ∧ FPSuccess.RegionFrame (call p).Out s final :=
  KernelExecution.complete rest p chosen s hm

#axiomsClean bmm_chunk_fwd_native
#axiomsClean native_launches
#axiomsClean bmm_chunk_fwd_if_local
#axiomsClean bmm_chunk_fwd_kernel
#axiomsClean bmm_chunk_fwd_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.TensorBridge
