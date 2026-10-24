import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.HostFrame

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Physical
noncomputable section

abbrev Byte := BitVec 8

/-- Storage-unit width belongs to the actual allocation. FP16 uses two bytes,
E4M3 uses one; no FP32-cell model is imposed on tensor storage. -/
structure Layout (σ : Type) where
  unit : Nat → Nat
  unitPositive : ∀ storage,0<unit storage
  capacity : σ → Nat → Nat
  observe : σ → Nat → Nat → Byte

def rawWord {σ : Type} (layout : Layout σ) (native : σ) (cell : Address) : BitVec 32 :=
  BitVec.ofNat 32 (∑ lane : Fin (layout.unit cell.1),
    (layout.observe native cell.1 (layout.unit cell.1*cell.2+lane.val)).toNat*256^lane.val)

def wordByte (word : BitVec 32) (lane : Nat) : Byte := BitVec.ofNat 8 (word.toNat/256^lane)

inductive Marker where
  | old (point : Address)
  | changed

def initialMarkers : Memory Marker := Marker.old

def arithmetic : Source.Arithmetic Marker Marker where
  inputZero := .changed
  accumulatorZero := .changed
  dot _ _ _ _ := fun _ _ => .changed
  cast _ _ := .changed

def Representation {σ : Type} (layout : Layout σ) (initial final : σ) (memory : Memory Marker) : Prop :=
  ∀ cell original,memory cell=.old original → ∀ lane,lane<layout.unit cell.1 →
    layout.unit original.1*original.2+lane<layout.capacity initial original.1 →
    layout.unit cell.1*cell.2+lane<layout.capacity final cell.1 ∧
      layout.observe final cell.1 (layout.unit cell.1*cell.2+lane)=
        layout.observe initial original.1 (layout.unit original.1*original.2+lane)

/-- Ordinary allocation/type registration, independent of computed values. -/
structure Library {σ : Type} (runtime : Host.Runtime σ) (layout : Layout σ) : Prop where
  allocation : ∀ rows columns profile device initial output allocated,
    runtime.empty rows columns profile device initial=some (output,allocated) →
    layout.unit output.storage=profile.bytes

def CallerFrame {σ : Type} (layout : Layout σ) (live : σ → Nat → Prop) (initial final : σ) : Prop :=
  ∀ storage,live initial storage → ∀ byte,byte<layout.capacity initial storage →
    byte<layout.capacity final storage ∧ layout.observe final storage byte=layout.observe initial storage byte

theorem caller_bytes {σ : Type} (layout : Layout σ) (runtime : Host.Runtime σ) (live : σ → Nat → Prop)
    (args : Host.Arguments) (initial : σ) (final : Host.World Marker σ) (output : Tensor)
    (e : Host.Completion arithmetic runtime live args ⟨initialMarkers,initial,[]⟩ final output)
    (representation : Representation layout initial final.native final.memory) : CallerFrame layout live initial final.native := by
  intro storage present byte bounded
  have width := layout.unitPositive storage
  have lane := Nat.mod_lt byte width
  have coordinate : layout.unit storage*(byte/layout.unit storage)+byte%layout.unit storage=byte := by
    simpa only [Nat.add_comm] using Nat.mod_add_div byte (layout.unit storage)
  have marker : final.memory (storage,byte/layout.unit storage)=.old (storage,byte/layout.unit storage) :=
    e.caller_frame _ present
  have bytes := representation _ _ marker (byte%layout.unit storage) lane
    (by simpa only [coordinate] using bounded)
  simpa only [coordinate] using bytes

#axiomsClean caller_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Physical
