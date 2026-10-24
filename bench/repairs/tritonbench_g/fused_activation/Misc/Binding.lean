import bench.repairs.tritonbench_g.fused_activation.Proofs.Graph
import bench.repairs.tritonbench_g.fused_activation.Proofs.Contract
import bench.repairs.tritonbench_g.fused_activation.Proofs.Pointers

namespace VeriTile.Bench.Repairs.FusedActivation.Binding
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Bytes := Nat → BitVec 8

def byteOf (word : BitVec 32) (byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 256^byte)

/-- Same x object and data pointer before and after; no fresh output exists. -/
def callerBytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (address : Nat) : BitVec 8 :=
  if g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel then
    byteOf (Graph.eval ops ((state.mem "x" ((address-g.dataPtr.toNat)/4)).read .real))
      ((address-g.dataPtr.toNat)%4)
  else before address

theorem coordinate_bytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (i b : Nat) (hi : i < g.numel) (hb : b < 4) :
    callerBytes ops g before state (g.dataPtr.toNat+4*i+b) =
      byteOf (Graph.eval ops ((state.mem "x" i).read .real)) b := by
  have range : g.dataPtr.toNat ≤ g.dataPtr.toNat+4*i+b ∧
      g.dataPtr.toNat+4*i+b < g.dataPtr.toNat+4*g.numel := by omega
  have index : (g.dataPtr.toNat+4*i+b-g.dataPtr.toNat)/4 = i := by omega
  have byte : (g.dataPtr.toNat+4*i+b-g.dataPtr.toNat)%4 = b := by omega
  simp only [callerBytes, if_pos range, index, byte]

/-- One source x store has precisely the ordinary four-byte store effect.
This lemma applies to arbitrary replacement words and arbitrary source state. -/
theorem store_bytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (i : Nat) (hi : i < g.numel) (v : Graph.Expr) (address : Nat) :
    callerBytes ops g before (state.write "x" i (.mk .real v)) address =
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

theorem exterior_bytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (address : Nat)
    (outside : address < g.dataPtr.toNat ∨ g.dataPtr.toNat+4*g.numel ≤ address) :
    callerBytes ops g before state address = before address := by
  have miss : ¬(g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel) := by omega
  simp [callerBytes,miss]

theorem initial_bytes (ops : Graph.Operations) (g : TensorMetadata) (before : Bytes)
    (state : State Graph.Expr) (xs : Nat → BitVec 32)
    (loads : ∀ j, j < g.numel → (state.mem "x" j).read .real = .leaf (xs j))
    (binding : ∀ j, j < g.numel → ∀ b, b < 4 →
      before (g.dataPtr.toNat+4*j+b) = byteOf (xs j) b) :
    callerBytes ops g before state = before := by
  funext address
  by_cases inside : g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel
  · let j := (address-g.dataPtr.toNat)/4
    let b := (address-g.dataPtr.toNat)%4
    have hj : j < g.numel := by dsimp [j]; omega
    have hb : b < 4 := by dsimp [b]; omega
    have eq : g.dataPtr.toNat+4*j+b = address := by dsimp [j,b]; omega
    rw [←eq,coordinate_bytes ops g before state j b hj hb,loads j hj]
    exact (binding j hj b hb).symm
  · simp [callerBytes,inside]

#axiomsClean coordinate_bytes
#axiomsClean store_bytes
#axiomsClean exterior_bytes
#axiomsClean initial_bytes
end
end VeriTile.Bench.Repairs.FusedActivation.Binding
