import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeInputs

/-! Task-local reuse of the ordinary typed-byte memory interpretation from the
fused-layernorm kernel proof. This models kernel buffer arguments and stores,
not allocation, streams, autotuning or any host return protocol. -/

namespace VeriTile.Bench.Repairs.LayerNormWelfold.Physical
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Bytes := Nat → BitVec 8

/-- The physical input reader assembles the four little-endian bytes at the
actual address. Its use neither copies nor changes the caller's memory. -/
def read32 (heap : Bytes) (address : Nat) : VeriTile.Torch.FP32.Word :=
  FP.Value.ofBits .fp32 ((heap address).toNat + 256*(heap (address+1)).toNat +
    65536*(heap (address+2)).toNat + 16777216*(heap (address+3)).toNat)

def byteOf (word : NativeKernel.Word) (byte : Nat) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 256^byte)

inductive Kind where
  | fp32 | bf16

def width : Kind → Nat | .fp32 => 4 | .bf16 => 2

def payload : Kind → Cell NativeKernel.Term → NativeKernel.Term
  | .fp32, c => c.read .real
  | .bf16, c => c.read .bf16

def tagged : Kind → NativeKernel.Term → Cell NativeKernel.Term
  | .fp32, v => .mk .real v
  | .bf16, v => .mk .bf16 v

/-- Typed regional-to-byte interpretation at an actual kernel output pointer. -/
def regionBytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (kind : Kind)
    (region : RegionName) (pointer : BitVec 64) (count : Nat) (before : Bytes)
    (state : State NativeKernel.Term) (address : Nat) : BitVec 8 :=
  if pointer.toNat ≤ address ∧ address < pointer.toNat + width kind * count then
    byteOf (NativeKernel.observe I P (payload kind (state.mem region ((address-pointer.toNat)/width kind))))
      ((address-pointer.toNat)%width kind)
  else before address

theorem coordinate_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (kind : Kind)
    (region : RegionName) (pointer : BitVec 64) (count : Nat) (before : Bytes)
    (state : State NativeKernel.Term) (i b : Nat) (hi : i < count) (hb : b < width kind) :
    regionBytes I P kind region pointer count before state (pointer.toNat + width kind*i+b) =
      byteOf (NativeKernel.observe I P (payload kind (state.mem region i))) b := by
  have range : pointer.toNat ≤ pointer.toNat+width kind*i+b ∧
      pointer.toNat+width kind*i+b < pointer.toNat+width kind*count := by cases kind <;> simp_all [width] <;> omega
  have index : (pointer.toNat+width kind*i+b-pointer.toNat)/width kind = i := by
    cases kind <;> simp_all [width] <;> omega
  have byte : (pointer.toNat+width kind*i+b-pointer.toNat)%width kind = b := by
    cases kind <;> simp_all [width] <;> omega
  simp only [regionBytes, if_pos range, index, byte]

/-- Every regional source store has exactly its ordinary two- or four-byte
physical effect. No final-output oracle is assumed by the byte interpretation. -/
theorem store_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (kind : Kind)
    (region : RegionName) (pointer : BitVec 64) (count : Nat) (before : Bytes)
    (state : State NativeKernel.Term) (i : Nat) (hi : i<count) (v : NativeKernel.Term) (address : Nat) :
    regionBytes I P kind region pointer count before (state.write region i (tagged kind v)) address =
      if pointer.toNat+width kind*i ≤ address ∧ address < pointer.toNat+width kind*i+width kind then
        byteOf (NativeKernel.observe I P v) (address-(pointer.toNat+width kind*i))
      else regionBytes I P kind region pointer count before state address := by
  by_cases inside : pointer.toNat ≤ address ∧ address < pointer.toNat+width kind*count
  · simp only [regionBytes, if_pos inside, State.write]
    by_cases same : (address-pointer.toNat)/width kind = i
    · have hit : pointer.toNat+width kind*i ≤ address ∧ address < pointer.toNat+width kind*i+width kind := by
        cases kind <;> simp_all [width] <;> omega
      have byte : (address-pointer.toNat)%width kind = address-(pointer.toNat+width kind*i) := by
        cases kind <;> simp_all [width] <;> omega
      cases kind <;> simp_all [tagged,payload]
    · have miss : ¬(pointer.toNat+width kind*i ≤ address ∧ address < pointer.toNat+width kind*i+width kind) := by
        cases kind <;> simp_all [width] <;> omega
      simp [same,miss]
  · have miss : ¬(pointer.toNat+width kind*i ≤ address ∧ address < pointer.toNat+width kind*i+width kind) := by
      cases kind <;> simp_all [width] <;> omega
    simp [regionBytes,inside,miss]

theorem exterior_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (kind : Kind)
    (region : RegionName) (pointer : BitVec 64) (count : Nat) (before : Bytes)
    (state : State NativeKernel.Term) (address : Nat)
    (outside : address < pointer.toNat ∨ pointer.toNat+width kind*count ≤ address) :
    regionBytes I P kind region pointer count before state address = before address := by
  have miss : ¬(pointer.toNat ≤ address ∧ address < pointer.toNat+width kind*count) := by omega
  simp [regionBytes,miss]

/-- Ordinary kernel output-buffer placement. Pointers are the actual launch
arguments; separation/capacity are allocation facts, not host execution goals. -/
structure Outputs (S D : Nat) where
  y : BitVec 64
  mean : BitVec 64
  rstd : BitVec 64
  yAligned : y.toNat % 2 = 0
  meanAligned : mean.toNat % 4 = 0
  rstdAligned : rstd.toNat % 4 = 0
  yFits : y.toNat + 2*(S*D) ≤ 2^64
  meanFits : mean.toNat + 4*S ≤ 2^64
  rstdFits : rstd.toNat + 4*S ≤ 2^64
  mean_y : mean.toNat+4*S ≤ y.toNat ∨ y.toNat+2*(S*D) ≤ mean.toNat
  rstd_y : rstd.toNat+4*S ≤ y.toNat ∨ y.toNat+2*(S*D) ≤ rstd.toNat
  mean_rstd : mean.toNat+4*S ≤ rstd.toNat ∨ rstd.toNat+4*S ≤ mean.toNat

def outputBytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (p : Outputs S D)
    (Y Mean Rstd : RegionName) (before : Bytes) (state : State NativeKernel.Term) : Bytes :=
  regionBytes I P .fp32 Rstd p.rstd S
    (regionBytes I P .fp32 Mean p.mean S
      (regionBytes I P .bf16 Y p.y (S*D) before state) state) state

theorem output_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (p : Outputs S D)
    (Y Mean Rstd : RegionName) (before : Bytes) (state : State NativeKernel.Term)
    (i b : Nat) (hi : i<S*D) (hb : b<2) :
    outputBytes I P p Y Mean Rstd before state (p.y.toNat+2*i+b) =
      byteOf (NativeKernel.observe I P ((state.mem Y i).read .bf16)) b := by
  unfold outputBytes
  rw [exterior_bytes, exterior_bytes]
  · exact coordinate_bytes I P .bf16 Y p.y (S*D) before state i b hi hb
  · rcases p.mean_y with h|h <;> simp only [width] <;> omega
  · rcases p.rstd_y with h|h <;> simp only [width] <;> omega

theorem mean_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (p : Outputs S D)
    (Y Mean Rstd : RegionName) (before : Bytes) (state : State NativeKernel.Term)
    (i b : Nat) (hi : i<S) (hb : b<4) :
    outputBytes I P p Y Mean Rstd before state (p.mean.toNat+4*i+b) =
      byteOf (NativeKernel.observe I P ((state.mem Mean i).read .real)) b := by
  unfold outputBytes
  rw [exterior_bytes]
  · exact coordinate_bytes I P .fp32 Mean p.mean S _ state i b hi hb
  · rcases p.mean_rstd with h|h <;> simp only [width] <;> omega

theorem rstd_bytes (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (p : Outputs S D)
    (Y Mean Rstd : RegionName) (before : Bytes) (state : State NativeKernel.Term)
    (i b : Nat) (hi : i<S) (hb : b<4) :
    outputBytes I P p Y Mean Rstd before state (p.rstd.toNat+4*i+b) =
      byteOf (NativeKernel.observe I P ((state.mem Rstd i).read .real)) b :=
  coordinate_bytes I P .fp32 Rstd p.rstd S _ state i b hi hb

def Outside (p : Outputs S D) (address : Nat) : Prop :=
  (address < p.y.toNat ∨ p.y.toNat+2*(S*D) ≤ address) ∧
  (address < p.mean.toNat ∨ p.mean.toNat+4*S ≤ address) ∧
  (address < p.rstd.toNat ∨ p.rstd.toNat+4*S ≤ address)

/-- Fresh output allocations do not overlap caller input backing bytes. This
allows every read-only alias between X, weight and bias. -/
def FreshInputs (p : Outputs S D) (W B X : VeriTile.Torch.FP32.Tensor) : Prop :=
  ∀ t, t = W ∨ t = B ∨ t = X → ∀ i, i < t.storage.elements → ∀ b, b < 4 →
    Outside p (t.storage.address i+b)

theorem frame (I : NativeKernel.Instructions) (P : NativeKernel.Profile) (p : Outputs S D)
    (Y Mean Rstd : RegionName) (before : Bytes) (state : State NativeKernel.Term)
    (address : Nat) (outside : Outside p address) :
    outputBytes I P p Y Mean Rstd before state address = before address := by
  unfold outputBytes
  rw [exterior_bytes I P .fp32 Rstd p.rstd S _ state address outside.2.2,
    exterior_bytes I P .fp32 Mean p.mean S _ state address outside.2.1,
    exterior_bytes I P .bf16 Y p.y (S*D) before state address outside.1]

/-- No native pointer wraps at any observed output byte. -/
theorem output_pointers (p : Outputs S D) :
    (∀ i b, i<S*D → b<2 → (BitVec.ofNat 64 (p.y.toNat+2*i+b)).toNat = p.y.toNat+2*i+b) ∧
    (∀ i b, i<S → b<4 → (BitVec.ofNat 64 (p.mean.toNat+4*i+b)).toNat = p.mean.toNat+4*i+b) ∧
    (∀ i b, i<S → b<4 → (BitVec.ofNat 64 (p.rstd.toNat+4*i+b)).toNat = p.rstd.toNat+4*i+b) := by
  have hy := p.yFits
  have hm := p.meanFits
  have hr := p.rstdFits
  refine ⟨?_,?_,?_⟩ <;> intro i b hi hb <;> apply Nat.mod_eq_of_lt <;> omega

#axiomsClean store_bytes
#axiomsClean output_bytes
#axiomsClean mean_bytes
#axiomsClean rstd_bytes
#axiomsClean frame
#axiomsClean output_pointers
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormWelfold.Physical
