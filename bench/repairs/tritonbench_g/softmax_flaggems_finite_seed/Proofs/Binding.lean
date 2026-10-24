import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.NativeHost
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.Contract

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Binding
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Bytes := Nat → BitVec 8

def byteOf (word : BitVec 32) (byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 256^byte)

/-- Byte interpretation of one output region at its allocated data pointer. -/
def callerBytes (I : Native.Instructions) (S : Native.Schedule) (region : RegionName) (g : TensorMetadata) (before : Bytes)
    (state : State Native.Term) (address : Nat) : BitVec 8 :=
  if g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel then
    byteOf (Native.eval I S ((state.mem region ((address-g.dataPtr.toNat)/4)).read .real))
      ((address-g.dataPtr.toNat)%4)
  else before address

theorem coordinate_bytes (I : Native.Instructions) (S : Native.Schedule) (region : RegionName) (g : TensorMetadata) (before : Bytes)
    (state : State Native.Term) (i b : Nat) (hi : i < g.numel) (hb : b < 4) :
    callerBytes I S region g before state (g.dataPtr.toNat+4*i+b) =
      byteOf (Native.eval I S ((state.mem region i).read .real)) b := by
  have range : g.dataPtr.toNat ≤ g.dataPtr.toNat+4*i+b ∧
      g.dataPtr.toNat+4*i+b < g.dataPtr.toNat+4*g.numel := by omega
  have index : (g.dataPtr.toNat+4*i+b-g.dataPtr.toNat)/4 = i := by omega
  have byte : (g.dataPtr.toNat+4*i+b-g.dataPtr.toNat)%4 = b := by omega
  simp only [callerBytes, if_pos range, index, byte]

/-- One source destination store has precisely the ordinary four-byte store effect.
This lemma applies to arbitrary replacement words and arbitrary source state. -/
theorem store_bytes (I : Native.Instructions) (S : Native.Schedule) (region : RegionName) (g : TensorMetadata) (before : Bytes)
    (state : State Native.Term) (i : Nat) (hi : i < g.numel) (v : Native.Term) (address : Nat) :
    callerBytes I S region g before (state.write region i (.mk .real v)) address =
      if g.dataPtr.toNat+4*i ≤ address ∧ address < g.dataPtr.toNat+4*i+4 then
        byteOf (Native.eval I S v) (address-(g.dataPtr.toNat+4*i))
      else callerBytes I S region g before state address := by
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

theorem exterior_bytes (I : Native.Instructions) (S : Native.Schedule) (region : RegionName) (g : TensorMetadata) (before : Bytes)
    (state : State Native.Term) (address : Nat)
    (outside : address < g.dataPtr.toNat ∨ g.dataPtr.toNat+4*g.numel ≤ address) :
    callerBytes I S region g before state address = before address := by
  have miss : ¬(g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel) := by omega
  simp [callerBytes,miss]

theorem initial_bytes (I : Native.Instructions) (S : Native.Schedule) (region : RegionName) (g : TensorMetadata) (before : Bytes)
    (state : State Native.Term) (xs : Nat → BitVec 32)
    (loads : ∀ j, j < g.numel → (state.mem region j).read .real = .leaf region j (xs j))
    (binding : ∀ j, j < g.numel → ∀ b, b < 4 →
      before (g.dataPtr.toNat+4*j+b) = byteOf (xs j) b) :
    callerBytes I S region g before state = before := by
  funext address
  by_cases inside : g.dataPtr.toNat ≤ address ∧ address < g.dataPtr.toNat+4*g.numel
  · let j := (address-g.dataPtr.toNat)/4
    let b := (address-g.dataPtr.toNat)%4
    have hj : j < g.numel := by dsimp [j]; omega
    have hb : b < 4 := by dsimp [b]; omega
    have eq : g.dataPtr.toNat+4*j+b = address := by dsimp [j,b]; omega
    rw [←eq,coordinate_bytes I S region g before state j b hj hb,loads j hj]
    exact (binding j hj b hb).symm
  · simp [callerBytes,inside]

#axiomsClean coordinate_bytes
#axiomsClean store_bytes
#axiomsClean exterior_bytes
#axiomsClean initial_bytes

/-- Ordinary freshness of the two simultaneously live allocator results. -/
def Separate (p y : TensorMetadata) : Prop :=
  p.dataPtr.toNat+4*p.numel≤y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤p.dataPtr.toNat

def returnedBytes (I : Native.Instructions) (S : Native.Schedule) (p y : TensorMetadata)
    (before : Bytes) (forward final : State Native.Term) : Bytes :=
  callerBytes I S "in_grad" y (callerBytes I S "out" p before forward) final

theorem returned_output (I : Native.Instructions) (S : Native.Schedule) (p y : TensorMetadata)
    (separate : Separate p y) (before : Bytes) (forward final : State Native.Term)
    (i b : Nat) (hi : i<p.numel) (hb : b<4) :
    returnedBytes I S p y before forward final (p.dataPtr.toNat+4*i+b)=
      byteOf (Native.eval I S ((forward.mem "out" i).read .real)) b := by
  unfold returnedBytes
  rw [exterior_bytes]
  · exact coordinate_bytes I S "out" p before forward i b hi hb
  · rcases separate with h|h <;> omega

theorem returned_gradient (I : Native.Instructions) (S : Native.Schedule) (p y : TensorMetadata)
    (before : Bytes) (forward final : State Native.Term)
    (i b : Nat) (hi : i<y.numel) (hb : b<4) :
    returnedBytes I S p y before forward final (y.dataPtr.toNat+4*i+b)=
      byteOf (Native.eval I S ((final.mem "in_grad" i).read .real)) b :=
  coordinate_bytes I S "in_grad" y _ final i b hi hb

theorem returned_frame (I : Native.Instructions) (S : Native.Schedule) (p y : TensorMetadata)
    (before : Bytes) (forward final : State Native.Term) (address : Nat)
    (hp : address<p.dataPtr.toNat ∨ p.dataPtr.toNat+4*p.numel≤address)
    (hy : address<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤address) :
    returnedBytes I S p y before forward final address=before address := by
  unfold returnedBytes
  rw [exterior_bytes I S "in_grad" y _ final address hy,exterior_bytes I S "out" p before forward address hp]

#axiomsClean returned_output
#axiomsClean returned_gradient
#axiomsClean returned_frame
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Binding
