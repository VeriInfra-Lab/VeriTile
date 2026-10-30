import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeAdmission
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Cleanup

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup.Native
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
open Classical
set_option maxHeartbeats 3000000

theorem geometry (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h) :
    pid0 p<2^31 ∧ 0<Host.cdiv h.shape.chunk 64 ∧
    Host.cdiv h.shape.chunk 64<2^63 ∧
    pid0 p/Host.cdiv h.shape.chunk 64<Host.cdiv h.shape.chunk 32 ∧
    pid0 p%Host.cdiv h.shape.chunk 64<Host.cdiv h.shape.chunk 64 := by
  have chunk := (Finite.positive_dimensions h.shape metadata.sizes).2.2.2
  have small := (Finite.dimensions_bounded h.shape metadata.sizes).2.1
  have gx := metadata.gridX
  have bound : pid0 p<Host.cdiv h.shape.chunk 32*Host.cdiv h.shape.chunk 64 :=
    (p ⟨0,by change 0<3; decide⟩).isLt
  have tiles : 0<Host.cdiv h.shape.chunk 64 := by
    have cover := Geometry.cdiv_covers (n:=h.shape.chunk) (b:=64) (by decide)
    by_contra zero
    have eq : Host.cdiv h.shape.chunk 64=0 := by omega
    rw [eq,Nat.zero_mul] at cover
    omega
  have ceil : Host.cdiv h.shape.chunk 64<2^63 := by
    unfold Host.cdiv
    have := Nat.div_le_self (h.shape.chunk+64-1) 64
    norm_num at small ⊢
    omega
  exact ⟨by change Host.cdiv h.shape.chunk 32*Host.cdiv h.shape.chunk 64≤2147483647 at gx; norm_num; omega,
    tiles,ceil,(Nat.div_lt_iff_lt_mul tiles).mpr bound,Nat.mod_lt _ tiles⟩

theorem row_word (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h) (i : Fin 32) :
    row h p i=word (Cleanup.row h p i) := by
  have g := geometry h metadata p
  have small := (Finite.dimensions_bounded h.shape metadata.sizes).2.1
  have ceil : (word h.shape.chunk+64-1).sdiv 64=word (Host.cdiv h.shape.chunk 64) :=
    NativeControl.ceil_exact h.shape.chunk 64 (by decide) (by omega)
  unfold row
  rw [ceil]
  simp only [word]
  rw [NativeControl.division _ _ (by have := g.1; norm_num at *; omega) (by have := g.2.2.1; norm_num at *; omega)]
  change (BitVec.ofNat 64 (pid0 p/Host.cdiv h.shape.chunk 64))*BitVec.ofNat 64 32+BitVec.ofNat 64 i.val=_
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  rfl

theorem column_word (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h) (j : Fin 64) :
    column h p j=word (Cleanup.column h p j) := by
  have g := geometry h metadata p
  have small := (Finite.dimensions_bounded h.shape metadata.sizes).2.1
  have ceil : (word h.shape.chunk+64-1).sdiv 64=word (Host.cdiv h.shape.chunk 64) :=
    NativeControl.ceil_exact h.shape.chunk 64 (by decide) (by omega)
  unfold column
  rw [ceil]
  simp only [word]
  rw [NativeControl.remainder _ _ (by have := g.1; norm_num at *; omega) (by have := g.2.2.1; norm_num at *; omega)]
  change (BitVec.ofNat 64 (pid0 p%Host.cdiv h.shape.chunk 64))*BitVec.ofNat 64 64+BitVec.ofNat 64 j.val=_
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  rfl

theorem lane_bounds (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h)
    (i : Fin 32) (j : Fin 64) : Cleanup.row h p i<2^63 ∧ Cleanup.column h p j<2^63 := by
  have g := geometry h metadata p
  have small := (Finite.dimensions_bounded h.shape metadata.sizes).2.1
  have rb := Finite.padded_index h.shape.chunk 32 _ i (by decide) g.2.2.2.1 i.isLt
  have cb := Finite.padded_index h.shape.chunk 64 _ j (by decide) g.2.2.2.2 j.isLt
  change pid0 p/Host.cdiv h.shape.chunk 64*32+i.val<2^63 ∧
    pid0 p%Host.cdiv h.shape.chunk 64*64+j.val<2^63
  norm_num at small ⊢
  omega

theorem mask_exact (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h)
    (i : Fin 32) (j : Fin 64) : mask h p i j=decide (active h p i j) := by
  have bounds := lane_bounds h metadata p i j
  have small := (Finite.dimensions_bounded h.shape metadata.sizes).2.1
  unfold mask
  rw [row_word h metadata,column_word h metadata]
  simp only [word,BitVec.slt_eq_decide,
    Finite.signed64 _ (by have := bounds.1; norm_num at this; omega),
    Finite.signed64 _ (by have := bounds.2; norm_num at this; omega),
    Finite.signed64 h.shape.chunk (by omega)]
  simp [active,Bool.and_assoc]

theorem offset_word (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h)
    (i : Fin 32) (j : Fin 64) : offset h p i j=word (Cleanup.offset h p i j) := by
  unfold offset
  rw [row_word h metadata,column_word h metadata]
  simp only [word,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  congr 1
  unfold Cleanup.offset
  ring

theorem offset_exact (h : Host.Call) (metadata : NativeControl.Metadata h config) (p : PID h)
    (i : Fin 32) (j : Fin 64) (valid : active h p i j) :
    (offset h p i j).toNat=Cleanup.offset h p i j := by
  rw [offset_word h metadata]
  simp only [word,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have bound := Cleanup.offset_bound h p i j valid
  have size := metadata.sizes.outputBytes
  norm_num at size ⊢
  omega

theorem addresses_exact (h : Host.Call) (metadata : NativeControl.Metadata h config) :
    addresses h=Cleanup.addresses h := by
  unfold addresses Cleanup.addresses
  apply List.flatMap_congr
  intro p _
  apply List.flatMap_congr
  intro i _
  apply List.filterMap_congr
  intro j _
  rw [mask_exact h metadata]
  by_cases valid : active h p i j
  · simp only [valid,decide_true,ite_true,offset_exact h metadata p i j valid]
  · simp [valid]

/-- The finite clearing program stores only the four bytes of an actual upper
output cell. Its pointer and the last byte cannot wrap. -/
theorem bytes_exact (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (nonempty : 0<(shape p).outputSize) (pid : PID (call p)) (i : Fin 32) (j : Fin 64)
    (valid : active (call p) pid i j) :
    NativePointers.Address (Direct.outputBuffer p h a) (offset (call p) pid i j) (Cleanup.offset (call p) pid i j) :=
  NativePointers.address _ _ _
    (offset_word (call p) (Direct.nativeMetadata p h config config_member nonempty) pid i j)
    (Cleanup.offset_bound (call p) pid i j valid)

#axiomsClean addresses_exact
#axiomsClean bytes_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Cleanup.Native
