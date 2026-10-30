import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.PhysicalMemory

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel.Bytes
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

abbrev Word := BitVec 32
abbrev Byte := BitVec 8
abbrev Store := Nat → Byte

/-- The same 32-bit carrier holds raw FP32 words and zero-extended narrow
words. The source Cell dtype determines whether a word is FP32/FP16/E5. -/
def read32 (s : Store) (a : Nat) : Word :=
  BitVec.ofNat 32 ((s a).toNat+256*(s (a+1)).toNat+65536*(s (a+2)).toNat+16777216*(s (a+3)).toNat)
def read16 (s : Store) (a : Nat) : Word :=
  BitVec.ofNat 32 ((s a).toNat+256*(s (a+1)).toNat)
def readE5 (s : Store) (a : Nat) : Word := BitVec.ofNat 32 (s a).toNat
def write (fp8 : Bool) (s : Store) (a : Nat) (value : Word) : Store :=
  fun j => if j=a then BitVec.ofNat 8 value.toNat
    else if !fp8 && decide (j=a+1) then BitVec.ofNat 8 (value.toNat/256) else s j
def storable (fp8 : Bool) (value : Word) : Prop :=
  value.toNat<2^(8*outputWidth fp8)

theorem byte_frame (fp8 : Bool) (s : Store) (a j : Nat) (value : Word)
    (outside : j<a ∨ a+outputWidth fp8≤j) : write fp8 s a value j=s j := by
  cases fp8 <;> simp [outputWidth] at outside
  all_goals have h0 : j≠a := by omega
  · have h1 : j≠a+1 := by omega
    simp [write,h0,h1]
  · simp [write,h0]

theorem read16_same (s : Store) (a : Nat) (value : Word) (valid : storable Bool.false value) :
    read16 (write Bool.false s a value) a=value := by
  have bound : value.toNat<65536 := valid
  have near : a+1≠a := by omega
  have digits : value.toNat%256+256*((value.toNat/256)%256)=value.toNat := by omega
  simp only [read16,write,ite_true,near,ite_false,Bool.not_false,decide_true,Bool.and_self,
    BitVec.toNat_ofNat]
  simp [digits]

theorem readE5_same (s : Store) (a : Nat) (value : Word) (valid : storable Bool.true value) :
    readE5 (write Bool.true s a value) a=value := by
  have bound : value.toNat<256 := valid
  simp [readE5,write,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]

theorem apart_byte (fp8 : Bool) (a b n k : Nat) (hk : k<n)
    (apart : Apart a (outputWidth fp8) b n) : b+k<a ∨ a+outputWidth fp8≤b+k := by
  unfold Apart at apart
  omega

def access : Access Word Store Byte where
  read32 := read32
  read16 := read16
  readE5 := readE5
  read8 := fun s a => s a
  byte := fun s a => s a
  write := write
  storable := storable
  write16_same := read16_same
  writeE5_same := readE5_same
  read32_apart := by
    intro fp8 s a b value apart
    have h0 := byte_frame fp8 s a (b+0) value (apart_byte fp8 a b 4 0 (by decide) apart)
    have h1 := byte_frame fp8 s a (b+1) value (apart_byte fp8 a b 4 1 (by decide) apart)
    have h2 := byte_frame fp8 s a (b+2) value (apart_byte fp8 a b 4 2 (by decide) apart)
    have h3 := byte_frame fp8 s a (b+3) value (apart_byte fp8 a b 4 3 (by decide) apart)
    simp only [Nat.add_zero] at h0
    simp only [read32,h0,h1,h2,h3]
  read16_apart := by
    intro fp8 s a b value apart
    have h0 := byte_frame fp8 s a (b+0) value (apart_byte fp8 a b 2 0 (by decide) apart)
    have h1 := byte_frame fp8 s a (b+1) value (apart_byte fp8 a b 2 1 (by decide) apart)
    simp only [Nat.add_zero] at h0
    simp only [read16,h0,h1]
  readE5_apart := by
    intro fp8 s a b value apart
    have h0 := byte_frame fp8 s a (b+0) value (apart_byte fp8 a b 1 0 (by decide) apart)
    simpa only [readE5,Nat.add_zero] using congrArg (fun byte => BitVec.ofNat 32 byte.toNat) h0
  read8_apart := by
    intro fp8 s a b value apart
    exact byte_frame fp8 s a b value (by simpa only [Nat.add_zero] using apart_byte fp8 a b 1 0 (by decide) apart)
  byte_apart := byte_frame

theorem narrow16_storable (value : BitVec 16) :
    access.storable Bool.false (BitVec.ofNat 32 value.toNat) := by
  have bound := value.isLt
  change (BitVec.ofNat 32 value.toNat).toNat<65536
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  exact bound

theorem narrowE5_storable (value : Byte) :
    access.storable Bool.true (BitVec.ofNat 32 value.toNat) := by
  have bound := value.isLt
  change (BitVec.ofNat 32 value.toNat).toNat<256
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  exact bound

#axiomsClean byte_frame
#axiomsClean read16_same
#axiomsClean readE5_same
#axiomsClean access
#axiomsClean narrow16_storable
#axiomsClean narrowE5_storable
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.PhysicalKernel.Bytes
