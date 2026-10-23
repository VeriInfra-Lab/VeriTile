import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Native
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.InputContract

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Native
open VeriTile.Triton FP.Structural
noncomputable section

variable {α β Store Byte : Type} [Inhabited α] {api : Memory.API β Store Byte}
  {embed : β → α} {r : Request} {input : RegionName} {s : State α}

/-- Metadata observations on the original input handle. The complete backing
is included by Preparation.inputCaller; no reachable-word surrogate is used. -/
structure Binding (call : Preparation api embed r input s) where
  metadata : VeriTile.Torch.TensorMetadata
  shape : metadata.shape=call.input.shape
  device : metadata.device.index=call.input.device
  pointer : 0<metadata.numel → metadata.dataPtr.toNat=call.input.base
  admitted : Contract.InputContract metadata ⟨Bool.true,r.position⟩

theorem size_bound (call : Preparation api embed r input s) : r.size≤2^62 := by
  have h := call.inputFits
  norm_num at *
  omega

theorem factors_bound (r : Request) (nonempty : 0<r.size) :
    0<r.batches ∧ 0<r.rows ∧ 0<r.columns ∧
    r.batches≤r.size ∧ r.rows≤r.size ∧ r.columns≤r.size ∧ r.rows*r.columns≤r.size := by
  have hb : 0<r.batches := by by_contra h; simp [Request.size,show r.batches=0 by omega] at nonempty
  have hm : 0<r.rows := by by_contra h; simp [Request.size,show r.rows=0 by omega] at nonempty
  have hk : 0<r.columns := by by_contra h; simp [Request.size,show r.columns=0 by omega] at nonempty
  have hbmr : r.batches≤r.batches*r.rows := Nat.le_mul_of_pos_right _ hm
  have hmbr : r.rows≤r.batches*r.rows := Nat.le_mul_of_pos_left _ hb
  have hbm : r.batches*r.rows≤r.size := Nat.le_mul_of_pos_right _ hk
  have hkall : r.columns≤r.size := Nat.le_mul_of_pos_left _ (Nat.mul_pos hb hm)
  have hmk : r.rows*r.columns≤r.size := by
    simpa [Request.size,Nat.mul_assoc] using Nat.le_mul_of_pos_left (r.rows*r.columns) hb
  exact ⟨hb,hm,hk,hbmr.trans hbm,hmbr.trans hbm,hkall,hmk⟩

theorem grid_bound (r : Request) (nonempty : 0<r.size) : grid r≤r.size := by
  have h := factors_bound r nonempty
  have hm : rowParts r≤r.rows := by unfold rowParts; omega
  have hk : parts r≤r.columns := by unfold parts; omega
  exact (Nat.mul_le_mul hm hk).trans h.2.2.2.2.2.2

theorem scheduled_coordinates (r : Request) (l : Location) (member : l∈schedule r) :
    row r l<rowParts r*2 ∧ evenColumn r l+1<parts r*1024 := by
  have hp := (mem_schedule r l).mp member |>.2
  have hd : 0<parts r := by by_contra h; simp [grid,show parts r=0 by omega] at hp
  have hm : program l/parts r<rowParts r :=
    (Nat.div_lt_iff_lt_mul hd).mpr (by simpa [grid] using hp)
  have hn := Nat.mod_lt (program l) hd
  have hmi := (lane l).1.isLt
  have hni := (lane l).2.1.isLt
  dsimp [TileIndex] at hmi hni
  constructor
  · unfold row; omega
  · unfold evenColumn; omega

theorem checked_index_abi (call : Preparation api embed r input s) (binding : Binding call) :
    IndexABI api embed r input s call := by
  have hshape : binding.metadata.shape=[r.batches,r.rows,r.columns] := binding.shape.trans call.inputShape
  have hcount : binding.metadata.numel=r.size := by simp [VeriTile.Torch.TensorMetadata.numel,hshape,Request.size,Nat.mul_assoc]
  have hpad : Contract.paddedRows binding.metadata=rowParts r*2 := by
    simp [Contract.paddedRows,hshape,rowParts,Nat.mul_comm]
  have hs := size_bound call
  refine ⟨?_,?_⟩
  · intro nonempty
    have h := factors_bound r nonempty
    have hg := grid_bound r nonempty
    norm_num at *
    omega
  · intro l member
    have nonempty : 0<r.size := by
      by_contra h
      have empty := schedule_empty r (by omega)
      simpa [empty] using member
    have h := factors_bound r nonempty
    have coords := scheduled_coordinates r l member
    have hm : rowParts r*2≤r.rows+1 := by unfold rowParts; omega
    have hk : parts r*1024≤r.columns+1023 := Nat.div_mul_le_self _ _
    have lo := binding.admitted.lower
    have hi := binding.admitted.padded (by simpa [hcount] using nonempty)
    rw [hpad] at hi
    dsimp only at lo hi
    refine ⟨by norm_num at *; omega,by norm_num at *; omega,by omega,by omega,?_⟩
    intro active
    have offset := offset_lt r _ _ _ active.1 active.2.1 active.2.2
    have xi := call.inputFits
    have yo := call.outputFits
    constructor
    · norm_num at *; omega
    · constructor <;> omega

theorem input_alignment (call : Preparation api embed r input s) (binding : Binding call)
    (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    logicalAddress call.input b m c%4=0 := by
  have nonempty : 0<binding.metadata.numel := by
    rw [VeriTile.Torch.TensorMetadata.numel,binding.shape,call.inputShape]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one]
    exact Nat.mul_pos (by omega) (Nat.mul_pos (by omega) (by omega))
  have h := binding.admitted.aligned nonempty
  rw [binding.pointer nonempty] at h
  rw [call.inputAddress b m c hb hm hc]
  simp [Nat.add_mod,h]

theorem input_backing_frame (api : Memory.API β Store Byte) (embed : β → α) (project : α → β)
    (ops : Operations α) (r : Request) (input output : RegionName) (s : State α)
    (call : Preparation api embed r input s) (offset : Nat) (bound : offset<call.input.storageBytes) :
    api.byte (entry api embed project ops r input output s call).memory (call.input.storageBase+offset)=
      api.byte call.before (call.input.storageBase+offset) := by
  exact (whole api embed project ops r input output s call).2.2.2.2 _
    (call.inputCaller _ (by omega) (by omega))

theorem native_addresses (call : Preparation api embed r input s) (binding : Binding call)
    (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    (BitVec.ofNat 64 (call.input.base+4*r.offset b m c)).toNat=logicalAddress call.input b m c ∧
    (BitVec.ofNat 64 (call.output.base+4*r.offset b m c)).toNat=logicalAddress call.output b m c ∧
    logicalAddress call.input b m c%4=0 ∧ logicalAddress call.output b m c%4=0 ∧
    logicalAddress call.input b m c+4≤call.input.allocationEnd ∧
    logicalAddress call.output b m c+4≤call.output.allocationEnd := by
  have offset := offset_lt r b m c hb hm hc
  have xi := call.inputFits
  have yo := call.outputFits
  have xb : call.input.base+4*r.offset b m c<2^64 := by omega
  have yb : call.output.base+4*r.offset b m c<2^64 := by omega
  rw [call.inputAddress b m c hb hm hc,call.outputAddress b m c hb hm hc]
  refine ⟨by simpa [BitVec.toNat_ofNat] using xb,
    by simpa [BitVec.toNat_ofNat] using yb,?_,?_,by omega,by omega⟩
  · simpa [call.inputAddress b m c hb hm hc] using input_alignment call binding b m c hb hm hc
  · simp [Nat.add_mod,call.outputAligned]

theorem original_input_spec (api : Memory.API ℝ Store Byte) (r : Request) (input : RegionName)
    (s : State ℝ) (call : Native.Preparation api id r input s)
    (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    Spec.result r (fun o => (s.mem input o).read .fp32) b m c=
      Spec.result r (fun o => api.read call.before (call.input.base+4*o)) b m c := by
  have hn : c/2*2<r.columns := lt_of_le_of_lt (Nat.div_mul_le_self c 2) hc
  have hu := call.inputBinding b m (c/2*2) hb hm hn
  by_cases hp : c/2*2+1<r.columns
  · have hv := call.inputBinding b m (c/2*2+1) hb hm hp
    simp [Spec.result,Spec.pair,hp,hu,hv]
  · simp [Spec.result,Spec.pair,hp,hu]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Native
