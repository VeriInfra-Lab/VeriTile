import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def signed32 (z : Int) := IntegerExpr.signedBits 32 (z % (4294967296 : Int)).toNat

theorem signed32_bitvec (z : Int) : signed32 z = (BitVec.ofInt 32 z).toInt :=
  signedBits_int_mod 32 z
def unsigned32 (z : Int) := (z % (4294967296 : Int)).toNat
def or32 (a b : Int) := IntegerExpr.signedBits 32 (unsigned32 a ||| unsigned32 b)

def sign (n : Nat) := n &&& 8
def positive (n : Nat) := n ^^^ sign n
def minusBias (n : Nat) := ((signed32 (↑(positive n >>> 1 : Nat) : Int) - 1) % (256 : Int)).toNat
def addedBias (n : Nat) := (minusBias n + 127) % 256
def shiftedExponent (n : Nat) := signed32 (signed32 (addedBias n) * 8388608)
def shiftedMantissa (n : Nat) := signed32 (signed32 (↑(positive n &&& 1 : Nat) : Int) * 4194304)
def normalWord (n : Nat) := or32 (shiftedExponent n) (shiftedMantissa n)
def zeroHandled (n : Nat) := if positive n=0 then 0 else normalWord n
def denormalHandled (n : Nat) := if positive n=1 then 1056964608 else zeroHandled n
def shiftedSign (n : Nat) := signed32 (signed32 (sign n) * 268435456)
/-- Exact lowered integer path, including two separate uint8 wraps and
signed32 carriers after shifts/OR. The final bitcast preserves its low32 bits. -/
def integerWord (n : Nat) : Int := or32 (denormalHandled n) (shiftedSign n)

def converted (n : Nat) : Word := embed (Bits.narrow ⟨BitVec.ofInt 32 (integerWord n)⟩)

/-- Closed finite-format check of every nibble, including negative zero. -/
theorem integer_payload : ∀ n : Bits.Nibble,
    BitVec.ofInt 32 (integerWord n.toNat) = Bits.sourceWord32 n := by
  decide +kernel

theorem byte_halves : ∀ b : Bits.Packed,
    (b.toNat >>> 4) = (Bits.high b).toNat ∧
    (b.toNat &&& 15) = (Bits.low b).toNat := by
  decide +kernel

theorem converted_table (n : Bits.Nibble) :
    converted n.toNat = embed ⟨Bits.expectedBits n⟩ := by
  unfold converted
  rw [integer_payload]
  have h := (Bits.all_nibbles n).2.1
  have hv : Bits.narrow ⟨Bits.sourceWord32 n⟩ = (⟨Bits.expectedBits n⟩ : FP.Value .bf16) := by
    exact congrArg (fun b : BitVec 16 => (⟨b⟩ : FP.Value .bf16)) h
  rw [hv]

#axiomsClean integer_payload
#axiomsClean byte_halves
#axiomsClean converted_table
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
