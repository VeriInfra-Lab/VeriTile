import bench.tritonbench_g.rope_transform.Contract

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

def memory {α β : Type} (embed : β→α) (read : Nat→β) (c : Call) : T.Memory α :=
  fun name offset=>.mk .fp32 (embed (read ((
    if name="Q" then (c.q Bool.false).dataPtr.toNat else
    if name="K" then (c.q Bool.true).dataPtr.toNat else
    if name="C" then (c.tables Bool.false).dataPtr.toNat else
    (c.tables Bool.true).dataPtr.toNat)+4*offset)))

def entry {α β : Type} [Inhabited α] (embed : β→α) (before : Nat→β) (c : Call) : State α :=
  {mem:=memory embed before c,regs:=fun _ _ _=>none,pids:=fun _=>0,
   numPids:=fun ax=>if ax=0 then c.geometry.rows else 1,undef:=fun d _ _=>defaultValue d}

theorem q_numel (c : Call) (admit : Admitted c) (k : Bool) :
    (c.q k).metadata.numel=RequestSize c.geometry k := by
  simp [VeriTile.Torch.TensorMetadata.numel,admit.qShape k,RequestSize,Request.rows,
    Request.stride,Nat.mul_assoc]

theorem table_numel (c : Call) (admit : Admitted c) (k : Bool) :
    (c.tables k).metadata.numel=c.geometry.sequence*c.geometry.half := by
  simp [VeriTile.Torch.TensorMetadata.numel,admit.tableShape k]

theorem contiguous_address (t : Tensor) (laws : Laws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (index : Nat) (bound : index<t.metadata.numel) :
    t.storage.address (t.physicalIndex index)=t.dataPtr.toNat+4*index := by
  have idx:=laws.contiguous contiguous index bound
  have cap:=laws.inBounds index bound
  rw [idx] at cap ⊢
  rw [Reuse.pointer_exact t laws (by omega) (by omega)]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

theorem q_address (c : Call) (admit : Admitted c) (k : Bool) (index : Nat)
    (bound : index<RequestSize c.geometry k) :
    (c.q k).storage.address ((c.q k).physicalIndex index)=(c.q k).dataPtr.toNat+4*index :=
  contiguous_address _ (c.qLaws k) (admit.qContiguous k) index (by rwa [q_numel c admit k])

theorem table_address (c : Call) (admit : Admitted c) (k : Bool) (index : Nat)
    (bound : index<c.geometry.sequence*c.geometry.half) :
    (c.tables k).storage.address ((c.tables k).physicalIndex index)=(c.tables k).dataPtr.toNat+4*index :=
  contiguous_address _ (c.tableLaws k) (admit.tableContiguous k) index (by rwa [table_numel c admit k])

theorem q_read {α β : Type} [Inhabited α] (embed : β→α) (before : Nat→β)
    (c : Call) (admit : Admitted c) (k : Bool) (index : Nat) (bound : index<RequestSize c.geometry k) :
    ((memory embed before c) (region k) index).read .fp32=embed (logicalRead before (c.q k) index) := by
  rw [logicalRead,q_address c admit k index bound]
  cases k <;> simp [memory,region]

theorem table_read {α β : Type} [Inhabited α] (embed : β→α) (before : Nat→β)
    (c : Call) (admit : Admitted c) (k : Bool) (index : Nat) (bound : index<c.geometry.sequence*c.geometry.half) :
    ((memory embed before c) (if k then "S" else "C") index).read .fp32=
      embed (logicalRead before (c.tables k) index) := by
  rw [logicalRead,table_address c admit k index bound]
  cases k <;> simp [memory]

def InputBinding (c : Call) (read : Nat→Word) : Prop :=
  (∀ k index,logicalRead read (c.q k) index=VeriTile.Torch.FP32.logicalValue read (c.q k) (c.qLaws k) index) ∧
  (∀ k index,logicalRead read (c.tables k) index=VeriTile.Torch.FP32.logicalValue read (c.tables k) (c.tableLaws k) index)

theorem input_binding (c : Call) (admit : Admitted c) (read : Nat→Word) : InputBinding c read :=
  ⟨fun k=>Reuse.logical_word read _ (c.qLaws k) (admit.qResolved k),
   fun k=>Reuse.logical_word read _ (c.tableLaws k) (admit.tableResolved k)⟩

#axiomsClean contiguous_address
#axiomsClean q_read
#axiomsClean table_read
#axiomsClean input_binding
end
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
