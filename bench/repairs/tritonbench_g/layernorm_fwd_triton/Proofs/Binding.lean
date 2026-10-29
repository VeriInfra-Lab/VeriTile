import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceCall
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.CallFP
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.CallReal

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.Wrapper
open Torch Address Memory Execution HostPrepared
open VeriTile.Bench.Repairs.LayerNormFwd.Address (byteAddress)
noncomputable section

theorem coordinate_positive (t : Tensor) (index : List Nat) (h : Torch.Coordinate t index) :
    0<t.shape.prod := by
  change List.Forall₂ (fun (i n : Nat) => i<n) index t.shape at h
  have each (is ns : List Nat) (valid : List.Forall₂ (fun (i n : Nat) => i<n) is ns) : 0<ns.prod := by
    induction valid with
    | nil => simp
    | @cons i n is ns hi hs ih => simpa using Nat.mul_pos (by omega : 0<n) ih
  exact each index t.shape h

theorem input_alignment (m : VeriTile.Torch.TensorMetadata) (t : Tensor)
    (binding : MetadataBinding m t) (h : Contract.TensorContract m)
    (index : List Nat) (valid : Torch.Coordinate t index) : address t index % 4=0 := by
  have positive : 0<m.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,binding.shape] using coordinate_positive t index valid
  have aligned := h.alignment positive
  rw [binding.pointer positive] at aligned
  have width := t.fp32Width (binding.dtype.mp h.dtype)
  simp [address,byteAddress,width,Nat.add_mod,aligned]

variable {α Store Byte Host : Type} {api : API α Store Byte} {neg : α → α} {caller : Nat → Prop}
  {platform : Runtime.Platform α} {service : Runtime.Service platform api neg caller Host}
  {lib : APIHost api neg caller Host} {q : Request} {initial : State Store Host}

/-- Inputs, pointer metadata and all nine scalar arguments come from the
original handles. No resolve, transfer, clone, contiguous or view route runs. -/
theorem direct_inputs (c : SuccessfulCall platform api neg service lib q initial) :
    c.setup.prepared.x=q.x ∧ c.setup.prepared.w=q.w ∧
    c.returned=c.setup.preparation.y ∧
    c.setup.prepared.xv.seqStride=q.x.strides.getD 0 0 ∧
    c.setup.prepared.xv.headStride=q.x.strides.getD 1 0 ∧
    c.setup.prepared.xv.dimStride=q.x.strides.getD 2 0 ∧
    c.setup.prepared.wv.headStride=q.w.strides.getD 0 0 ∧
    c.setup.prepared.wv.dimStride=q.w.strides.getD 1 0 := by
  have xs := c.setup.prepared.xv.strides
  have ws := c.setup.prepared.wv.strides
  change q.x.strides=[_,_,_] at xs
  change q.w.strides=[_,_] at ws
  refine ⟨rfl,rfl,rfl,?_,?_,?_,?_,?_⟩ <;> simp [xs,ws]

theorem call_native_addresses (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Fin c.setup.shape.columns) :
    nativePointer c.setup.prepared.layout.x row.1.val row.2.val col.val
        c.setup.prepared.strides.xSeq c.setup.prepared.strides.xHead c.setup.prepared.strides.xDim=
      address q.x (HostLayout.index3 row col) ∧
    nativePointer c.setup.prepared.layout.w 0 row.2.val col.val 0
        c.setup.prepared.strides.wHead c.setup.prepared.strides.wDim=
      address q.w (HostLayout.index2 row col) ∧
    nativePointer c.setup.prepared.layout.y row.1.val row.2.val col.val
        c.setup.prepared.strides.ySeq c.setup.prepared.strides.yHead c.setup.prepared.strides.yDim=
      address c.returned (HostLayout.index3 row col) := by
  exact ⟨(native_x _ _ _).trans (HostPrepared.x_address _ _ _),
    (native_w _ _ _).trans (HostPrepared.w_address _ _ _),
    (native_y _ _ _).trans (HostPrepared.y_address _ _ _)⟩

theorem call_aligned (c : SuccessfulCall platform api neg service lib q initial)
    (row : Row c.setup.shape) (col : Fin c.setup.shape.columns) :
    address q.x (HostLayout.index3 row col) % 4=0 ∧
    address q.w (HostLayout.index2 row col) % 4=0 ∧
    address c.returned (HostLayout.index3 row col) % 4=0 := by
  refine ⟨input_alignment _ _ q.xObserved c.setup.domain.admitted.input _ (call_x_coordinate c row col),
    input_alignment _ _ q.wObserved c.setup.domain.admitted.weight _ (call_w_coordinate c row col),?_⟩
  have aligned := lib.allocate_aligned q.x c.setup.enterState c.setup.preparation.y
    c.setup.preparation.allocateY ⟨_,HostLayout.coordinate3 c.setup.prepared.yv row col⟩
  have width := c.setup.prepared.y4
  change c.returned.allocation.width=4 at width
  change c.returned.allocation.base%4=0 at aligned
  simp [address,byteAddress,width,Nat.add_mod,aligned]

theorem call_storage_frame (c : SuccessfulCall platform api neg service lib q initial) :
    (∀ i, q.xStorage.backing.base ≤ i → i < q.xStorage.backing.base+q.xStorage.backing.nbytes →
      api.byte c.final.memory i=api.byte initial.memory i) ∧
    (∀ i, q.wStorage.backing.base ≤ i → i < q.wStorage.backing.base+q.wStorage.backing.nbytes →
      api.byte c.final.memory i=api.byte initial.memory i) :=
  ⟨fun i hlo hhi => call_frame c i (c.setup.domain.xBacking i hlo hhi),
    fun i hlo hhi => call_frame c i (c.setup.domain.wBacking i hlo hhi)⟩

theorem call_shape_dtype (c : SuccessfulCall platform api neg service lib q initial) :
    c.returned.shape=q.x.shape ∧ c.returned.dtype=0 ∧ c.returned.device=q.x.device := by
  have info := lib.allocate_metadata q.x c.setup.enterState c.setup.preparation.y c.setup.preparation.allocateY
  exact ⟨info.1,(setup_dtype c.setup).2.2,info.2.2.1⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.Wrapper
