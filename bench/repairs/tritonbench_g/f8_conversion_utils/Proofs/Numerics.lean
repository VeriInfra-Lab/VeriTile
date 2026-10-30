import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.Bytes

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Numerics
noncomputable section
open Bytes
set_option maxRecDepth 24000

/-- Signed magnitude keeps the sign of zero independently of its Real value.
NaNs retain their class; exact payload policy is separately configuration-bound. -/
inductive ScalarValue where
  | finite (negative : Bool) (magnitude : ℚ)
  | infinity (negative : Bool)
  | nan (negative : Bool) (payload : Nat)

def wordNat (width : Nat) (word : Word width) : Nat :=
  ((List.finRange width).map fun k => (word k).toNat*256^k.val).sum

/-- Independent IEEE binary-format interpretation; this does not use a
Triton cast, Torch conversion, or the source kernel output. -/
def ieee (exponentBits fractionBits bias bits : Nat) : ScalarValue :=
  let sign := decide (bits / 2^(exponentBits+fractionBits) % 2=1)
  let exponent := bits / 2^fractionBits % 2^exponentBits
  let fraction := bits % 2^fractionBits
  if exponent=2^exponentBits-1 then
    if fraction=0 then .infinity sign else .nan sign fraction
  else
    let magnitude : ℚ := if exponent=0 then
      fraction*(2 : ℚ)^((1 : Int)-bias-fractionBits)
    else (2^fractionBits+fraction : Nat)*(2 : ℚ)^((exponent : Int)-bias-fractionBits)
    .finite sign magnitude

def e5 (word : Word 1) : ScalarValue := ieee 5 2 15 (word 0).toNat
def half (word : Word 2) : ScalarValue := ieee 5 10 15 (wordNat 2 word)
def single (word : Word 4) : ScalarValue := ieee 8 23 127 (wordNat 4 word)

def e5NaN (byte : BitVec 8) : Prop := byte.toNat/4%32=31 ∧ byte.toNat%4≠0
instance (byte : BitVec 8) : Decidable (e5NaN byte) := inferInstanceAs (Decidable (_ ∧ _))

/-- For all finite E5 values and both infinities, extension to half is exact:
the eight source bits occupy the high byte of IEEE FP16. -/
def decodeE5 (nanPolicy : Word 1 → Word 2) (word : Word 1) : Word 2 :=
  if e5NaN (word 0) then nanPolicy word else fun k => if k.val=0 then 0 else word 0

def positiveE5 (code : Nat) : ℚ :=
  let exponent := code/4
  let fraction := code%4
  if exponent=0 then fraction/65536
  else (4+fraction : Nat)*(2 : ℚ)^((exponent : Int)-17)

/-- Exact rational nearest-even selection among finite positive E5 values.
Restricting the candidate set to 0..123 implements the selected native
satfinite policy, including values beyond the overflow midpoint. -/
def nearer (magnitude : ℚ) (best candidate : Nat) : Nat :=
  let old := |positiveE5 best-magnitude|
  let next := |positiveE5 candidate-magnitude|
  if next<old then candidate
  else if next=old ∧ candidate%2=0 then candidate else best

def nearestFiniteE5 (magnitude : ℚ) : Nat :=
  (List.range 124).foldl (nearer magnitude) 0

def signedByte (negative : Bool) (code : Nat) : Word 1 :=
  fun _ => BitVec.ofNat 8 ((if negative then 128 else 0)+code)

def encodeE5 (nanPolicy : ScalarValue → Word 1) (value : ScalarValue) : Word 1 :=
  match value with
  | .finite sign magnitude => signedByte sign (nearestFiniteE5 magnitude)
  | .infinity sign => signedByte sign 123
  | .nan _ _ => nanPolicy value

/-- NaN class is fixed independently of the native conversion. Only the
configured sign/payload choice is supplied; finite outputs cannot masquerade
as an unresolved NaN policy. -/
def ScalarValue.IsNaN : ScalarValue → Prop
  | .nan _ _ => True
  | _ => False

structure NaNPolicy where
  decode : Word 1 → Word 2
  encode : ScalarValue → Word 1
  decode_class : ∀ word,e5NaN (word 0) → (half (decode word)).IsNaN
  encode_class : ∀ value,value.IsNaN → (e5 (encode value)).IsNaN

/-- Optional primitive conformance facts for another backend interpretation.
These are modeling/compiler facts, not task-specific numerical experiments.
The current public bit-machine theorem defines its casts by `reference`. -/
structure DefaultE5Casts
    (decode : Word 1 → Word 2) (encodeHalf : Word 2 → Word 1) (encodeSingle : Word 4 → Word 1)
    (nanPolicy : NaNPolicy) : Prop where
  decode_cast : ∀ word,decode word=decodeE5 nanPolicy.decode word
  half_cast : ∀ word,encodeHalf word=encodeE5 nanPolicy.encode (half word)
  single_cast : ∀ word,encodeSingle word=encodeE5 nanPolicy.encode (single word)

/-- A compatibility relation for a separately supplied primitive backend.
The current public theorem has no configured-cast hypothesis. -/
def ConfiguredCast (physical reference : Word iw → Word ow) : Prop :=
  ∀ input,physical input=reference input

/-- NVIDIA E4M3 finite encoding has one NaN magnitude code and no infinity. -/
def e4NaN (byte : BitVec 8) : Prop := byte.toNat % 128 = 127
instance (byte : BitVec 8) : Decidable (e4NaN byte) := inferInstanceAs (Decidable (_ = _))

def e4 (word : Word 1) : ScalarValue :=
  let bits := (word 0).toNat
  let sign := decide (128 ≤ bits)
  let exponent := bits/8%16
  let fraction := bits%8
  if bits%128=127 then .nan sign fraction
  else .finite sign (if exponent=0 then fraction/512
    else (8+fraction : Nat)*(2 : ℚ)^((exponent : Int)-10))

def halfWord (bits : Nat) : Word 2 :=
  fun k => BitVec.ofNat 8 (bits / 256^k.val)

/-- Independent exact E4-to-half extension, including E4 subnormals.
The sign bit is preserved, including both zeros. NaNs use the selected policy. -/
def decodeE4 (nanPolicy : Word 1 → Word 2) (word : Word 1) : Word 2 :=
  let bits := (word 0).toNat
  let exponent := bits/8%16
  let fraction := bits%8
  let sign := bits/128*32768
  if e4NaN (word 0) then nanPolicy word
  else halfWord (sign + if exponent=0 then
    [0,6144,7168,7680,8192,8448,8704,8960][fraction]!
    else (exponent+8)*1024+fraction*128)

def positiveE4 (code : Nat) : ℚ :=
  let exponent := code/8
  let fraction := code%8
  if exponent=0 then fraction/512
  else (8+fraction : Nat)*(2 : ℚ)^((exponent : Int)-10)

def nearerE4 (magnitude : ℚ) (best candidate : Nat) : Nat :=
  let old := |positiveE4 best-magnitude|
  let next := |positiveE4 candidate-magnitude|
  if next<old then candidate
  else if next=old ∧ candidate%2=0 then candidate else best

def encodeE4 (nanPolicy : ScalarValue → Word 1) (value : ScalarValue) : Word 1 :=
  match value with
  | .finite sign magnitude => signedByte sign ((List.range 127).foldl (nearerE4 magnitude) 0)
  | .infinity sign => signedByte sign 126
  | .nan _ _ => nanPolicy value

structure NaNPolicyE4 where
  decode : Word 1 → Word 2
  encode : ScalarValue → Word 1
  decode_class : ∀ word,e4NaN (word 0) → (half (decode word)).IsNaN
  encode_class : ∀ value,value.IsNaN → (e4 (encode value)).IsNaN

structure Policy where
  e5 : NaNPolicy
  e4 : NaNPolicyE4

/-- An explicit inhabited semantic policy: canonical positive quiet NaNs.
Other hardware sign/payload policies can be modeled by another `Policy`.
Neither choice excludes any input representation from the kernel domain. -/
def canonicalPolicy : Policy where
  e5 := {
    decode := fun _ => halfWord 32256
    encode := fun _ => signedByte Bool.false 127
    decode_class := by intro _ _; change True; trivial
    encode_class := by intro _ _; change True; trivial }
  e4 := {
    decode := fun _ => halfWord 32256
    encode := fun _ => signedByte Bool.false 127
    decode_class := by intro _ _; change True; trivial
    encode_class := by intro _ _; change True; trivial }

/-- Six independent scalar references: round-to-nearest-even, satfinite for
encode, exact finite decode, and an explicitly class-preserving NaN policy. -/
def reference (policy : Policy) (p : Source.Profile) : Word p.inputWidth → Word p.outputWidth :=
  match p with
  | .decodeE5 => decodeE5 policy.e5.decode
  | .encodeHalfE5 => fun word => encodeE5 policy.e5.encode (half word)
  | .encodeSingleE5 => fun word => encodeE5 policy.e5.encode (single word)
  | .decodeE4 => decodeE4 policy.e4.decode
  | .encodeHalfE4 => fun word => encodeE4 policy.e4.encode (half word)
  | .encodeSingleE4 => fun word => encodeE4 policy.e4.encode (single word)

/-- Primitive-model compatibility retained for older callers. This is not a
numerical optimization law and is not an assumption of the public theorem. -/
def LocalCast (policy : Policy) (p : Source.Profile)
    (native : Word p.inputWidth → Word p.outputWidth) : Prop :=
  ∀ word,native word=reference policy p word

#specNonCircular reference avoiding [Source.kernel]

end
end VeriTile.Bench.Repairs.F8ConversionUtils.Numerics
