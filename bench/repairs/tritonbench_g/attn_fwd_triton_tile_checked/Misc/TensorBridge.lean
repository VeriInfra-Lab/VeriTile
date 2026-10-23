import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Correct
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.RawReal
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.RawNative
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedBinding
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.CheckedNative
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Tensor
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Proofs.Address

/-! Complete Tensor/application statements retained from the public entries.
The actual source grids are supplied by the raw kernel theorems. -/

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.TensorBridge
open VeriTile Triton AttnFwdTriton Checked
open Verified Verified.PhysicalCells
open VeriTile.Bench.Repairs.LayerNormFwd
set_option maxHeartbeats 1000000

/-- Complete causal Real result from the caller memory and observed input
views. Logical input agreement and trace bounds are derived internally. -/
theorem whole_real (p : RealProofs.Parameters) (a : Layout) (A : FlatAlloc)
    (base : Role→Nat) (views : Views a A p base)
    (q k v qs ks : TensorMetadata) (hinput : inputChecks (realGeometry p) q k v qs ks=Bool.true)
    (hl : launchChecks (realGeometry p) qs ks=Bool.true)
    (hp : pointerChecks (realGeometry p) base=Bool.true)
    (bounds : RegionBounds) (storage : Storage p A bounds qs ks)
    (memory : Nat→ℝ) (template : BlockState)
    (stage : p.stage=3) (nonoverlap : OutputLayout.NonOverlapping p)
    (fresh : FreshBytes a (GridFootprint.window A p)) :
    inputChecks (realGeometry p) q k v qs ks=Bool.true ∧
    NativeControl.CudaLaunchable p (HostShape.blockD p.headDim) ∧
    (∀ r n, n<envelope (realGeometry p) → base r%width r=0 ∧ base r+width r*n+width r<2^64) ∧
    ∃ t, ∃ final : AliasMemory.Physical a.toLayout,
      Nonempty (Kernel.GridLaunchedOrdinary (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false)
        (GridCoordinates.grid p) (AliasMemory.pull a.toLayout (RealBinding.cells a memory) template) t) ∧
      (∀ i : OutputLayout.Coordinate p,
        Host.readReal (final (Host.outputCell a A p base views i))=
          RealProofs.expected p.heads p.context p.headDim i.1.val i.2.1.val i.2.2.1.val i.2.2.2.val (RealBinding.inputs p base memory)) ∧
      (∀ idx : GridIndex (GridCoordinates.grid p),
        Stmt.TraceSafeList bounds (GridLaunch.kernel A p (HostShape.blockD p.headDim) Bool.false).body
          ((AliasMemory.pull a.toLayout (RealBinding.cells a memory) template).withGridIndex idx)) ∧
      (∀ cell : Fin a.capacity,
        (∀ k, GridFootprint.window A p k → a.address k≠cell.val) → final cell=(RealBinding.cells a memory) cell) := by
  refine ⟨hinput,Checked.launchable (realGeometry p) qs ks (inherited _ _ _ hl),?_,?_⟩
  · intro r n hn
    have h:=formed_pointer (realGeometry p) base hp r n hn
    exact ⟨h.1,h.2.1⟩
  · exact VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RawReal.whole_real p a A base views bounds
      (checked_windows p A bounds qs ks storage (inherited _ _ _ hl))
      memory template stage nonoverlap fresh

#axiomsClean whole_real
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.TensorBridge

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.TensorBridge
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton TypedExecution Checked
open VeriTile.Bench.Repairs.LayerNormFwd
open TypedPhysical (Physical pull)
open VeriTile.Bench.Repairs.LayerNormFwd.Address (signed64)
open Verified.PhysicalCells (Layout Role FreshBytes width)
open Graph
noncomputable section
set_option maxHeartbeats 2000000

theorem inputs_resolved (p : Config) (objects : Role→TensorDescriptor)
    (hinput : inputChecks (typedGeometry p) (objects .query).metadata (objects .key).metadata
      (objects .value).metadata (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (r : Role) (hr : r≠Role.output) : (objects r).metadata.negative=Bool.false := by
  simp only [inputChecks,Bool.and_eq_true,List.all_eq_true] at hinput
  have hm : (objects r).metadata∈[(objects .query).metadata,(objects .key).metadata,
      (objects .value).metadata,(objects .queryScale).metadata,(objects .keyScale).metadata] := by
    cases r <;> simp_all
  have hc:=hinput.1 _ hm
  simp only [commonChecks,decide_eq_true_eq] at hc
  exact hc.2.2.2.2.1

/-- Complete production division graph from actual input bytes to the actual
BF16 returned descriptor. The value specification is the independent two-loop
recurrence, including TF32 score dot, FP16 value dot result, and both FMAs. -/
theorem whole_native (ops : Operations) (p : Config) (a : Layout)
    (objects : Role→TensorDescriptor) (views : Views a p (pointers objects))
    (tensors : Role→Torch.FP32.Tensor) (tensorViews : TensorBinding.View p objects tensors)
    (memory : ByteMemory) (template : State Word)
    (hinput : inputChecks (typedGeometry p) (objects .query).metadata (objects .key).metadata
      (objects .value).metadata (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (hl : launchChecks (typedGeometry p) (objects .queryScale).metadata (objects .keyScale).metadata=Bool.true)
    (hp : pointerChecks (typedGeometry p) (pointers objects)=Bool.true)
    (binding : OutputBinding p objects) (stage : p.stage=3) (nonoverlap : NonOverlapping p)
    (fresh : FreshBytes a (fun k=>Window p k.1 k.2)) (allocationFresh : FreshInputs objects) :
    (∀ shape ∈ tileShapes (HostShape.blockD p.headDim),shape.prod≤1048576) ∧
    NativeLaunch.CudaLaunchable (launchShape p) (HostShape.blockD p.headDim) ∧
    ControlCertificate p (HostShape.blockD p.headDim) ∧
    (∀ z h n d r,roleActive p z h n d r →
      machineOffset p z h n d r=roleOffset p z h n d r ∧
      signed64 (machineOffset p z h n d r)=(roleOffset p z h n d r : Int) ∧
      Verified.NativeOffsets.bytePointer (pointers objects r) (width r) (machineOffset p z h n d r)=
        a.address (roleKey p z h n d r)) ∧
    (∀ r,r≠Role.output → (objects r).dtype=.fp32) ∧
    (∀ z h n d r,r≠Role.output → roleActive p z h n d r →
      ((Physical.initial a memory template).mem (roleKey p z h n d r).1
        (roleKey p z h n d r).2).read .real=Expr.leaf (descriptorRead objects memory r z h n d)) ∧
    (∀ z h n d r,∀ hr : r≠Role.output,roleActive p z h n d r →
      ((Physical.initial a memory template).mem (roleKey p z h n d r).1
        (roleKey p z h n d r).2).read .real=
          Expr.leaf (TensorBinding.logicalRead p memory tensors tensorViews r hr z h n d)) ∧
    (∀ r n,n<envelope (typedGeometry p) →
      (objects r).dataPointer%width r=0 ∧ (objects r).dataPointer+width r*n+width r<2^64) ∧
    ∃ final, ∃ graph : Physical Expr a.toLayout, ∃ physical : Physical Word a.toLayout,
    ∃ result : ReturnedBytes,
      Nonempty (GridRun backend p (HostShape.blockD p.headDim) Bool.false
        (Physical.initial a memory template) final) ∧
      pull a.toLayout graph final=final ∧ physical=Physical.evaluatePhysical ops graph ∧
      result=finishBytes p objects
        (fun c=>BitVec.ofNat 16 ((physical (outputCell a p (pointers objects) views c)).read .bf16).toNat) memory ∧
      result.tensor=objects .output ∧
      result.tensor.dtype=.bf16 ∧ result.tensor.metadata.shape=[p.batches,p.heads,p.context,p.headDim] ∧
      result.tensor.metadata.strides=((typedGeometry p).stride .output).raw ∧
      result.tensor.metadata.device=(objects .query).metadata.device ∧
      (∀ c : Coordinate p,result.read p c=BitVec.ofNat 16
        ((Physical.reference ops p a memory template c).read .bf16).toNat) ∧
      (∀ c : Coordinate p,∃ w,physical (outputCell a p (pointers objects) views c)=Cell.mk .bf16 w ∧
        result.read p c=BitVec.ofNat 16 w.toNat) ∧
      (∀ address : Fin (2^64),(∀ i,address.val≠outputByte p objects i) →
        result.memory address=memory address) ∧
      (∀ r,r≠Role.output → ∀ address : Fin (2^64),
        (objects r).allocationBase≤address.val →
        address.val<(objects r).allocationBase+(objects r).allocationBytes →
        result.memory address=memory address) ∧
      (∀ cell : Fin a.capacity,(∀ k,Window p k.1 k.2 → a.address k≠cell.val) →
        physical cell=Physical.callerCells a memory cell) := by
  obtain ⟨final,graph,physical,hgrid,hpull,heval,hread,hframe⟩ :=
    VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.RawNative.whole_native ops p a (pointers objects) views memory template stage nonoverlap fresh
  let word : Coordinate p→BF16Word := fun c=>BitVec.ofNat 16
    ((physical (outputCell a p (pointers objects) views c)).read .bf16).toNat
  let result:=finishBytes p objects word memory
  refine ⟨admitted_tiles _ _ _ hl,Checked.launchable _ _ _ (inherited _ _ _ hl),
    Address.control p _ _ hl,?_,input_dtype p objects hinput,?_,?_,?_,final,graph,physical,result,hgrid,hpull,heval,rfl,rfl,
    binding.dtype,binding.shape,binding.strides,binding.device,?_,?_,?_,?_,hframe⟩
  · exact fun z h n d r ha=>Address.native_active p a (pointers objects) views
      (objects .queryScale).metadata (objects .keyScale).metadata hl hp z h n d r ha
  · exact fun z h n d r hr ha=>Physical.actual_input p a objects views memory template hinput z h n d r hr ha
  · intro z h n d r hr ha
    rw [Physical.actual_input p a objects views memory template hinput z h n d r hr ha,
      TensorBinding.descriptor_logical p objects tensors tensorViews memory r hr
        (inputs_resolved p objects hinput r hr) z h n d ha
        (TensorBinding.valid_index p objects tensors tensorViews hinput (inherited _ _ _ hl) r hr z h n d ha)]
  · intro r n hn
    have h:=formed_pointer (typedGeometry p) (pointers objects) hp r n hn
    exact ⟨h.1,h.2.1⟩
  · intro c
    rw [finish_read p objects nonoverlap binding word memory hp c]
    unfold word
    rw [hread c]
  · intro c
    have hw : ∃ w,Physical.reference ops p a memory template c=Cell.mk .bf16 w := by
      exact ⟨_,rfl⟩
    obtain ⟨w,hw⟩:=hw
    refine ⟨w,(hread c).trans hw,?_⟩
    rw [finish_read p objects nonoverlap binding word memory hp c]
    unfold word
    rw [hread c,hw]
    rfl
  · exact fun address h=>finish_frame p objects word memory address h
  · exact fun r hr address hlo hhi=>finish_inputs_frame p objects binding allocationFresh
      word memory r hr address hlo hhi

theorem empty_return (p : Config) (objects : Role→TensorDescriptor) (memory : ByteMemory)
    (h : p.batches*p.heads*p.context*p.headDim=0) :
    IsEmpty (Coordinate p) ∧ (finishBytes p objects (fun _=>0) memory).tensor=objects .output ∧
      (finishBytes p objects (fun _=>0) memory).memory=memory :=
  empty_native_return p objects memory h

#axiomsClean whole_native
#axiomsClean empty_return
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.TensorBridge
