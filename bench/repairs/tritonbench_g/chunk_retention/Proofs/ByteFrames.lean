import bench.repairs.tritonbench_g.chunk_retention.Proofs.ByteStores

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000

theorem finish_preserves_allocation (c : Shape) (objects : Region→Tensor) (events : List Address)
    (checks : StoreAdmission c objects events) (after : Heap Word) (before : ByteMemory)
    (t : Tensor) (fresh : ∀ a∈events,Disjoint (objects a.region) t) (point : Fin (2^64))
    (hlo : t.storageBase≤point.val) (hhi : point.val<t.storageBase+t.storageBytes) :
    finishBytes c objects events after before point=before point := by
  apply finish_frame
  intro i hi he
  have hb:=byte_bounds c objects events checks i hi
  have hf:=fresh i.1 ((byteEvents_mem events i.1 i.2).mp hi)
  unfold Disjoint at hf
  omega

theorem read_preserved (extent : Nat) (t : Tensor) (hp : pointerChecks extent t=Bool.true)
    (before after : ByteMemory)
    (frame : ∀ point : Fin (2^64),t.storageBase≤point.val → point.val<t.storageBase+t.storageBytes →
      after point=before point) (n : Nat) (hn : n<t.numel) :
    readWord after (t.dataPointer+4*n)=readWord before (t.dataPointer+4*n) := by
  unfold readWord
  congr 1
  funext j
  have hb:=active_bytes extent t hp n hn j.val j.isLt
  simp only [extendMemory,dif_pos hb.2.2]
  exact frame ⟨_,hb.2.2⟩ hb.1 hb.2.1

/-- The source's original zeros_like branch, before any backward load. -/
def zeroTensor (t : Tensor) (before : ByteMemory) : ByteMemory := fun point=>
  if t.dataPointer≤point.val ∧ point.val<t.dataPointer+4*t.numel then 0 else before point

theorem zero_read (extent : Nat) (t : Tensor) (hp : pointerChecks extent t=Bool.true)
    (before : ByteMemory) (n : Nat) (hn : n<t.numel) :
    readWord (zeroTensor t before) (t.dataPointer+4*n)=0 := by
  unfold readWord
  have he : (fun j : Fin 4=>extendMemory (zeroTensor t before) (t.dataPointer+4*n+j.val))=fun _=>0 := by
    funext j
    have hj:=j.isLt
    have hb:=active_bytes extent t hp n hn j.val hj
    simp only [extendMemory,dif_pos hb.2.2,zeroTensor]
    rw [if_pos (by constructor <;> omega)]
  rw [he]
  rfl

theorem zero_frame (t : Tensor) (before : ByteMemory) (point : Fin (2^64))
    (outside : ¬(t.dataPointer≤point.val ∧ point.val<t.dataPointer+4*t.numel)) :
    zeroTensor t before point=before point := if_neg outside

theorem zero_preserves_allocation (extent : Nat) (t saved : Tensor) (hp : pointerChecks extent t=Bool.true)
    (fresh : Disjoint t saved) (before : ByteMemory) (point : Fin (2^64))
    (hlo : saved.storageBase≤point.val) (hhi : point.val<saved.storageBase+saved.storageBytes) :
    zeroTensor t before point=before point := by
  apply zero_frame
  intro h
  simp only [pointerChecks,decide_eq_true_eq] at hp
  have he:=t.pointer
  rw [hp.1] at he
  unfold Disjoint at fresh
  omega

#axiomsClean finish_preserves_allocation
#axiomsClean read_preserved
#axiomsClean zero_read
#axiomsClean zero_frame
#axiomsClean zero_preserves_allocation
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
