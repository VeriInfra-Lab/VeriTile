import bench.tritonbench_g.rope_backward_transform.Contract

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

/-- All source regions read the same physical entry memory, including aliases
between the two readonly tables. -/
def memory {α β : Type} (embed : β→α) (read : Nat→β) (c : Call) : Source.Memory α :=
  fun name offset=>.mk .fp32 (embed (read ((
    if name="Q" then (c.q Bool.false).dataPtr.toNat else
    if name="K" then (c.q Bool.true).dataPtr.toNat else
    if name="C" then (c.tables Bool.false).dataPtr.toNat else
    (c.tables Bool.true).dataPtr.toNat)+4*offset)))

def logicalRead (read : Nat→β) (tensor : Tensor) (index : Nat) : β :=
  read (tensor.storage.address (tensor.physicalIndex index))

theorem logical_word (read : Nat→Word) (t : Tensor) (laws : Laws t)
    (resolved : t.metadata.isNeg=Bool.false) (index : Nat) :
    logicalRead read t index=VeriTile.Torch.FP32.logicalValue read t laws index := by
  simp [logicalRead,VeriTile.Torch.FP32.logicalValue,resolved]

theorem q_numel (c : Call) (admit : Admitted c) (k : Bool) :
    (c.q k).metadata.numel=c.geometry.size k := by
  simp [VeriTile.Torch.TensorMetadata.numel,admit.qShape k,Request.size,Request.rows,Nat.mul_assoc]

theorem table_numel (c : Call) (admit : Admitted c) (k : Bool) :
    (c.tables k).metadata.numel=c.geometry.tableRows k*c.geometry.half := by
  simp [VeriTile.Torch.TensorMetadata.numel,admit.tableShape k]

theorem pointer_exact (t : Tensor) (laws : Laws t) (positive : 0<t.metadata.numel)
    (offset : t.storageOffset<t.storage.elements) :
    t.dataPtr.toNat=t.storage.address t.storageOffset := by
  have p := congrArg BitVec.toNat (laws.pointer positive)
  have span := t.storage.spanFits
  have bound : t.storage.address t.storageOffset<2^64 := by
    unfold VeriTile.Torch.FP32.Storage.address
    omega
  simpa only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using p

theorem q_address (c : Call) (admit : Admitted c) (k : Bool) (index : Nat)
    (bound : index<c.geometry.size k) :
    (c.q k).storage.address ((c.q k).physicalIndex index)=(c.q k).dataPtr.toNat+4*index := by
  have h : index<(c.q k).metadata.numel := by rw [q_numel c admit k]; exact bound
  have indexLaw := (c.qLaws k).contiguous (admit.contiguous k) index h
  have cap := (c.qLaws k).inBounds index h
  rw [indexLaw] at cap ⊢
  rw [pointer_exact _ (c.qLaws k) (by omega) (by omega)]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem table_address (c : Call) (admit : Admitted c) (k : Bool) (row lane : Nat)
    (hr : row<c.geometry.tableRows k) (hl : lane<c.geometry.half) :
    (c.tables k).storage.address ((c.tables k).physicalIndex (row*c.geometry.half+lane))=
      (c.tables k).dataPtr.toNat+4*(row*c.geometry.rowStride k+lane*c.geometry.colStride k) := by
  have bound : row*c.geometry.half+lane<(c.tables k).metadata.numel := by
    rw [table_numel c admit k]
    have prod := Nat.mul_le_mul_right c.geometry.half (show row+1≤c.geometry.tableRows k by omega)
    simp only [Nat.add_mul,Nat.one_mul] at prod
    omega
  have idx := admit.tableIndex k row lane hr hl
  have cap := (c.tableLaws k).inBounds _ bound
  rw [idx] at cap ⊢
  rw [pointer_exact _ (c.tableLaws k) (by omega) (by omega)]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem q_read {α β : Type} [Inhabited α] (embed : β→α) (before : Nat→β)
    (c : Call) (admit : Admitted c) (k : Bool) (index : Nat) (bound : index<c.geometry.size k) :
    ((memory embed before c) (region k) index).read .fp32=embed (logicalRead before (c.q k) index) := by
  rw [logicalRead,q_address c admit k index bound]
  cases k <;> simp [memory,region]

theorem table_read {α β : Type} [Inhabited α] (embed : β→α) (before : Nat→β)
    (c : Call) (admit : Admitted c) (k : Bool) (row lane : Nat)
    (hr : row<c.geometry.tableRows k) (hl : lane<c.geometry.half) :
    ((memory embed before c) (if k then "S" else "C")
      (row*c.geometry.rowStride k+lane*c.geometry.colStride k)).read .fp32=
      embed (logicalRead before (c.tables k) (row*c.geometry.half+lane)) := by
  rw [logicalRead,table_address c admit k row lane hr hl]
  cases k <;> simp [memory]

def InputBinding (c : Call) (read : Nat→Word) : Prop :=
  (∀ k index,logicalRead read (c.q k) index=VeriTile.Torch.FP32.logicalValue read (c.q k) (c.qLaws k) index) ∧
  (∀ k index,logicalRead read (c.tables k) index=VeriTile.Torch.FP32.logicalValue read (c.tables k) (c.tableLaws k) index)

theorem input_binding (c : Call) (admit : Admitted c) (read : Nat→Word) : InputBinding c read :=
  ⟨fun k=>logical_word read _ (c.qLaws k) (admit.qResolved k),
   fun k=>logical_word read _ (c.tableLaws k) (admit.tableResolved k)⟩

#axiomsClean q_address
#axiomsClean table_address
#axiomsClean q_read
#axiomsClean table_read
#axiomsClean input_binding
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
