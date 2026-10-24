import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.HostFrame
import bench.repairs.tritonbench_g.matmul_persistent_triton.Proofs.Physical

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Physical
noncomputable section

-- Reuse the reviewed generic storage-byte and marker interpretations.
abbrev Byte := MatmulPersistentTriton.Physical.Byte
abbrev Layout := MatmulPersistentTriton.Physical.Layout
abbrev rawWord := @MatmulPersistentTriton.Physical.rawWord
abbrev wordByte := MatmulPersistentTriton.Physical.wordByte
abbrev Marker := MatmulPersistentTriton.Physical.Marker
abbrev initialMarkers := MatmulPersistentTriton.Physical.initialMarkers
abbrev Representation := @MatmulPersistentTriton.Physical.Representation
abbrev CallerFrame := @MatmulPersistentTriton.Physical.CallerFrame

def arithmetic : Source.Arithmetic Marker Marker where
  inputZero := .changed
  accumulatorZero := .changed
  dot _ _ _ _ := fun _ _ => .changed
  activation _ _ := .changed
  cast _ _ := .changed

/-- Ordinary allocation/type registration, independent of computed values. -/
structure Library {σ : Type} (runtime : Host.Runtime σ) (layout : Layout σ) : Prop where
  allocation : ∀ rows columns profile device initial output allocated,
    runtime.empty rows columns profile device initial=some (output,allocated) →
    layout.unit output.storage=profile.bytes

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
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Physical
