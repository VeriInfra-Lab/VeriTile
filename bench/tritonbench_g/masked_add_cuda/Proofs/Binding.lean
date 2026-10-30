import bench.tritonbench_g.masked_add_cuda.Proofs.Contract
import bench.tritonbench_g.masked_add_cuda.Proofs.Graph
import bench.tritonbench_g.masked_add_cuda.Proofs.Indices

/-! Concrete little-endian byte observations of the in-place source memory.
The old caller byte function supplies every exterior byte. Inside the checked
contiguous grad window, bytes are observations of the source's stored FP32 word.
The local store theorem proves this representation changes exactly four bytes.
No whole-kernel memory postcondition is a premise. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.Binding
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Bytes := Nat → BitVec 8

def byteOf {width : Nat} (word : BitVec width) (byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 256^byte)

/-- Native Tensor load binding on one actual backing byte memory. The storage
span facts are ordinary valid-Tensor facts, not extra admission guards. -/
structure Inputs (g p m : TensorMetadata) (before : Bytes)
    (grad data : Nat → BitVec 32) (mask : Nat → BitVec (8*maskBytes (selectedMask m))) : Prop where
  gradSpan : g.dataPtr.toNat + 4*g.numel ≤ 2^64
  dataSpan : p.dataPtr.toNat + 4*g.numel ≤ 2^64
  maskSpan : m.dataPtr.toNat + maskBytes (selectedMask m)*g.numel ≤ 2^64
  gradBytes : ∀ i, i < g.numel → ∀ b, b < 4 →
    before (g.dataPtr.toNat+4*i+b) = byteOf (grad i) b
  dataBytes : ∀ i, i < g.numel → ∀ b, b < 4 →
    before (p.dataPtr.toNat+4*i+b) = byteOf (data i) b
  maskBytes : ∀ i, i < g.numel → ∀ b, b < maskBytes (selectedMask m) →
    before (m.dataPtr.toNat+maskBytes (selectedMask m)*i+b) = byteOf (mask i) b

/-- The bounds mask and every active source pointer use the same extents,
widths and observed pointers as admission and the byte binding. -/
theorem checked_addresses (g p m : TensorMetadata) (a : AlphaArgument)
    (h : InputContract g p m a) (before : Bytes)
    (grad data : Nat → BitVec 32) (mask : Nat → BitVec (8*maskBytes (selectedMask m)))
    (binding : Inputs g p m before grad data mask)
    (pid lane : Nat) (hp : pid < Indices.gridCount g.numel) (hl : lane < 1024)
    (active : pid*1024+lane < g.numel) :
    (Indices.pointer g.dataPtr.toNat 4 pid lane).toNat = g.dataPtr.toNat+4*(pid*1024+lane) ∧
    (Indices.pointer p.dataPtr.toNat 4 pid lane).toNat = p.dataPtr.toNat+4*(pid*1024+lane) ∧
    (Indices.pointer m.dataPtr.toNat (maskBytes (selectedMask m)) pid lane).toNat =
      m.dataPtr.toNat+maskBytes (selectedMask m)*(pid*1024+lane) := by
  rcases h with ⟨_, _, _, _, _, _, _, _, bound, _⟩
  have width : 0 < maskBytes (selectedMask m) := by
    cases selectedMask m <;> decide
  exact ⟨(Indices.pointer_exact _ 4 _ _ _ (by decide) bound hp hl active binding.gradSpan).1,
    (Indices.pointer_exact _ 4 _ _ _ (by decide) bound hp hl active binding.dataSpan).1,
    (Indices.pointer_exact _ _ _ _ _ width bound hp hl active binding.maskSpan).1⟩

/-- Signed integer-to-int1 casting tests nonzero, including minimum signed
integers; it does not take the low bit of the mask word. -/
theorem signed_zero_iff (width : Nat) (word : BitVec width) : word.toInt = 0 ↔ word = 0 := by
  rw [←BitVec.toInt_zero (w := width), BitVec.toInt_inj]
  rfl

/-- Same grad object and data pointer before and after; no fresh output exists. -/
def callerBytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (address : Nat) : BitVec 8 :=
  if g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel then
    byteOf (Graph.eval ops ((state.mem "grad" ((address-g.dataPtr.toNat)/4)).read .real))
      ((address-g.dataPtr.toNat)%4)
  else before address

theorem coordinate_bytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (i b : Nat) (hi : i < g.numel) (hb : b < 4) :
    callerBytes ops g before state (g.dataPtr.toNat+4*i+b) =
      byteOf (Graph.eval ops ((state.mem "grad" i).read .real)) b := by
  have range : g.dataPtr.toNat ≤ g.dataPtr.toNat+4*i+b ∧
      g.dataPtr.toNat+4*i+b < g.dataPtr.toNat+4*g.numel := by omega
  have index : (g.dataPtr.toNat+4*i+b-g.dataPtr.toNat)/4 = i := by omega
  have byte : (g.dataPtr.toNat+4*i+b-g.dataPtr.toNat)%4 = b := by omega
  simp only [callerBytes, if_pos range, index, byte]

/-- The physical representation agrees with the caller's initial backing. -/
theorem initial_bytes (ops : Graph.Operations) (g p m : TensorMetadata) (before : Bytes)
    (grad data : Nat → BitVec 32) (mask : Nat → BitVec (8*maskBytes (selectedMask m)))
    (binding : Inputs g p m before grad data mask) (initial : State Graph.Expr)
    (hg : ∀ i, i < g.numel → (initial.mem "grad" i).read .real = .leaf (grad i)) :
    callerBytes ops g before initial = before := by
  funext address
  by_cases inside : g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel
  · let i := (address-g.dataPtr.toNat)/4
    let b := (address-g.dataPtr.toNat)%4
    have hi : i < g.numel := by dsimp [i]; omega
    have hb : b < 4 := by dsimp [b]; omega
    have addr : g.dataPtr.toNat+4*i+b = address := by dsimp [i,b]; omega
    rw [←addr, coordinate_bytes ops g before initial i b hi hb, hg i hi]
    exact (binding.gradBytes i hi b hb).symm
  · simp [callerBytes, inside]

/-- One source grad store has precisely the ordinary four-byte store effect.
This lemma applies to arbitrary replacement words and arbitrary source state. -/
theorem store_bytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (i : Nat) (hi : i < g.numel) (v : Graph.Expr) (address : Nat) :
    callerBytes ops g before (state.write "grad" i (.mk .real v)) address =
      if g.dataPtr.toNat+4*i ≤ address ∧ address < g.dataPtr.toNat+4*i+4 then
        byteOf (Graph.eval ops v) (address-(g.dataPtr.toNat+4*i))
      else callerBytes ops g before state address := by
  by_cases inside : g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel
  · simp only [callerBytes, if_pos inside, State.write]
    by_cases same : (address-g.dataPtr.toNat)/4 = i
    · have hit : g.dataPtr.toNat+4*i ≤ address ∧ address < g.dataPtr.toNat+4*i+4 := by omega
      have byte : (address-g.dataPtr.toNat)%4 = address-(g.dataPtr.toNat+4*i) := by omega
      simp [same, hit, byte]
    · have miss : ¬(g.dataPtr.toNat+4*i ≤ address ∧ address < g.dataPtr.toNat+4*i+4) := by omega
      simp [same, miss]
  · have miss : ¬(g.dataPtr.toNat+4*i ≤ address ∧ address < g.dataPtr.toNat+4*i+4) := by omega
    simp [callerBytes, inside, miss]

/-- Every byte outside the actual active write intervals is unchanged,
including vetoed grad bytes, storage-offset prefixes, suffixes and input bytes. -/
theorem byte_frame (ops : Graph.Operations) (g p m : TensorMetadata) (before : Bytes)
    (grad data : Nat → BitVec 32) (mask : Nat → BitVec (8*maskBytes (selectedMask m)))
    (binding : Inputs g p m before grad data mask) (initial final : State Graph.Expr)
    (hg : ∀ i, i < g.numel → (initial.mem "grad" i).read .real = .leaf (grad i))
    (hm : ∀ i, i < g.numel → (initial.mem "mask" i).read .int = (mask i).toInt)
    (frame : ∀ r i, (r ≠ "grad" ∨ g.numel ≤ i ∨ (initial.mem "mask" i).read .int ≠ 0) →
      final.mem r i = initial.mem r i)
    (address : Nat)
    (outside : ∀ i, i < g.numel → (mask i).toInt = 0 →
      address < g.dataPtr.toNat+4*i ∨ g.dataPtr.toNat+4*i+4 ≤ address) :
    callerBytes ops g before final address = before address := by
  rw [←congrFun (initial_bytes ops g p m before grad data mask binding initial hg) address]
  by_cases inside : g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel
  · have hi : (address-g.dataPtr.toNat)/4 < g.numel := by omega
    have veto : (initial.mem "mask" ((address-g.dataPtr.toNat)/4)).read .int ≠ 0 := by
      rw [hm _ hi]
      intro zero
      have := outside _ hi zero
      omega
    simp only [callerBytes, if_pos inside, frame "grad" _ (Or.inr (Or.inr veto))]
  · simp [callerBytes, inside]

#axiomsClean coordinate_bytes
#axiomsClean checked_addresses
#axiomsClean signed_zero_iff
#axiomsClean initial_bytes
#axiomsClean store_bytes
#axiomsClean byte_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.Binding
