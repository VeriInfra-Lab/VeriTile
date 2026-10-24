import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Bytes
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.Observation
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def ReadByBytes {M α β : Type} (ops : Ops M α β) : Prop :=
  ∀ a b key,(∀ i : Fin 4,ops.observe a key.owner (key.address+i.val)=
    ops.observe b key.owner (key.address+i.val)) → ops.read a key=ops.read b key

theorem bytes_read : ReadByBytes Bytes.ops := by
  intro a b key same
  change Bytes.pack (fun i => a key.owner (key.address+i.val))=
    Bytes.pack (fun i => b key.owner (key.address+i.val))
  apply congrArg Bytes.pack
  funext i
  exact same i

/-- Real semantics is a shadow observation model, not an encoding of arbitrary
real numbers as four finite bytes. Observations hold a value at every address;
the concrete FP specialization instead uses the proven four-byte Word adapter. -/
abbrev Shadow (α : Type) := Nat → Nat → α

def shadowWrite (s : Shadow α) (key : Key) (value : α) : Shadow α :=
  fun owner address => if key.covers owner address then value else s owner address

def shadowOps : Ops (Shadow α) α α where
  read := fun s key => s key.owner key.address
  write := shadowWrite
  observe := id
  read_same := by intro s key value; simp [shadowWrite,Key.covers]
  read_other := by
    intro s writer value observed separate
    apply if_neg
    exact Key.disjoint_bytes separate observed.owner observed.address ⟨rfl,by omega,by omega⟩
  frame := by intro s writer value owner address outside; exact if_neg outside

theorem shadow_read : ReadByBytes (shadowOps (α:=α)) := by
  intro a b key same
  simpa only [shadowOps,Nat.add_zero] using same 0

def realNegation : Negation ℝ := ⟨Neg.neg,neg_neg⟩

#axiomsClean bytes_read
#axiomsClean shadow_read
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.Observation
