import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.Logical

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.Words
open VeriTile.Triton FP.Structural Execution
noncomputable section
set_option maxHeartbeats 4000000

abbrev F := BitVec 32
abbrev H := BitVec 16
abbrev Byte := BitVec 8
def embed (h : H) : F := BitVec.ofNat 32 h.toNat
def half (f : F) : H := BitVec.ofNat 16 f.toNat
@[simp] theorem half_embed (h : H) : half (embed h)=h := by
  apply BitVec.eq_of_toNat_eq
  simp [half,embed,h.isLt]

/-- Raw scalar instruction ports. Half operands/results and E5M2 payloads
are separate bit widths. There is no loop, reduction, matrix or output port. -/
structure Instructions where
  literal : ℝ → F
  fromInt : Int → F
  add : F → F → F
  mul : F → F → F
  div : F → F → F
  neg : F → F
  fma : F → F → F → F
  rsqrt : F → F
  sigmoidNeg : F → F
  mulFTZ : F → F → F
  mulHalf : H → H → H
  widen : H → F
  narrow : F → H
  e5Half : Byte → H
  e5Single : Byte → F
  mma16 : (Fin 16 → H) → (Fin 16 → H) → F → F

def byte (x : F) : Byte := BitVec.ofNat 8 x.toNat
def cast (a : Instructions) (src dst : FloatDType) (x : F) : F :=
  match src,dst with
  | .f8e5,.fp16 => embed (a.e5Half (byte x))
  | .f8e5,_ => a.e5Single (byte x)
  | .fp16,.fp16 => x
  | .fp16,_ => a.widen (half x)
  | _,.fp16 => embed (a.narrow x)
  | _,_ => x
def binary (a : Instructions) (dtype : FloatDType) (op : Binary) (x y : F) : F :=
  if dtype=.fp16 then embed (a.mulHalf (half x) (half y)) else
    match op with
    | .add => a.add x y | .mul => a.mul x y | .div => a.div x y
    | .sub => a.add x (a.neg y) | _ => 0
def square (a : Instructions) (x : F) := a.mulFTZ (a.literal 1) (a.mulFTZ x x)
def arithmetic (a : Instructions) : Arithmetic F where
  literal := fun _ _ r => a.literal r
  binary := fun _ => binary a
  unary := fun _ op x => match op with
    | .rsqrt => a.rsqrt x
    | .sigmoid => a.sigmoidNeg (a.neg x)
    | .neg _ => a.neg x
    | _ => 0
  cast := fun _ => cast a
  fromInt := fun _ => a.fromInt
  powi := fun _ x n => if n=2 then square a x else 0
  decode := fun _ b => BitVec.ofNat 32 b.val
  mma16 := fun x y acc => a.mma16 (fun i => half (x i)) (fun i => half (y i)) acc
def gates (a : Instructions) : GateInstructions F := ⟨a.neg,a.fma,a.sigmoidNeg⟩

/-- All 65,536 half patterns remain exactly representable in the carrier. -/
theorem half_representation_injective : Function.Injective embed := by
  intro x y h
  simpa using congrArg half h

abbrev Memory := RegionName → Nat → Byte
def decode16 (lo hi : Byte) : H := BitVec.ofNat 16 (lo.toNat+256*hi.toNat)
def encode16 (h : H) (i : Fin 2) : Byte := BitVec.ofNat 8 (h.toNat/256^i.val)
def read16 (mem : Memory) (r : RegionName) (o : Nat) : H := decode16 (mem r (2*o)) (mem r (2*o+1))

theorem decode_encode (h : H) : decode16 (encode16 h 0) (encode16 h 1)=h := by
  apply BitVec.eq_of_toNat_eq
  have bound := h.isLt
  simp [decode16,encode16]
  omega
theorem encode_decode (lo hi : Byte) (i : Fin 2) :
    encode16 (decode16 lo hi) i=if i.val=0 then lo else hi := by
  have hl := lo.isLt
  have hh := hi.isLt
  fin_cases i <;> apply BitVec.eq_of_toNat_eq <;> simp [encode16,decode16] <;> omega

def isByteRegion (p : Parameters) (r : RegionName) : Bool :=
  p.byte && decide (r="RMS" ∨ r="W1" ∨ r="W3")
def entry (p : Parameters) (mem : Memory) : State F where
  mem := fun r o => if isByteRegion p r then .mk .int (mem r o).toInt
    else .mk .fp16 (embed (read16 mem r o))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem entry_bytes (p : Parameters) (mem : Memory) : ByteValues (entry p mem) := by
  intro r hr o
  unfold entry
  change Float8E5Bits.isInt8 ((if isByteRegion p r then Cell.mk .int (mem r o).toInt
    else Cell.mk .fp16 (embed (read16 mem r o))).read .int)=Bool.true
  split
  · simp only [Cell.read_same,Float8E5Bits.isInt8,decide_eq_true_eq]
    constructor
    · exact BitVec.le_toInt _
    · exact BitVec.toInt_lt
  · simp [Cell.read,defaultValue,Float8E5Bits.isInt8]

def commit (mem : Memory) (final : State F) : Memory := fun r b =>
  if r="OUT" then encode16 (half ((final.mem "OUT" (b/2)).read .fp16)) ⟨b%2,by omega⟩
  else mem r b

theorem commit_entry (p : Parameters) (mem : Memory) : commit mem (entry p mem)=mem := by
  funext r b
  by_cases h : r=("OUT" : RegionName)
  · subst r
    simp only [commit,ite_true,entry,isByteRegion,show ("OUT" : RegionName)≠"RMS" by decide,
      show ("OUT" : RegionName)≠"W1" by decide,show ("OUT" : RegionName)≠"W3" by decide,
      false_or,decide_false,Bool.and_false,Bool.false_eq_true,ite_false,Cell.read_same,half_embed,read16,encode_decode]
    split <;> congr 1 <;> omega
  · simp [commit,h]

#axiomsClean half_representation_injective
#axiomsClean decode_encode
#axiomsClean encode_decode
#axiomsClean entry_bytes
#axiomsClean commit_entry
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.Words
