import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Math
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000

/-- Fixed K32 instruction families. WGMMA N is one of the actual hardware
tile widths; this is never a variable-length reduction port. -/
inductive WgmmaN where
  | n8 | n16 | n32 | n64 | n128 | n256
  deriving DecidableEq
inductive ISA where
  | mmaM16N8K32Satfinite
  | wgmmaM64K32 (n : WgmmaN)
  deriving DecidableEq

structure Ops (I F H : Type) where
  izero : I
  fzero : F
  hzero : H
  product8 : BitVec 8 → BitVec 8 → I
  iadd : I → I → I
  mma32 : ISA → (Fin 32 → BitVec 8) → (Fin 32 → BitVec 8) → I → I
  toFloat : I → F
  mul : F → F → F
  narrow : F → H
  hadd : H → H → H

inductive Micro (BK : Nat) where
  | scalar (k : Fin BK)
  | instruction (kind : ISA) (lanes : Fin 32 → Option (Fin BK))

def Micro.terms : Micro BK → List (Fin BK)
  | .scalar k => [k]
  | .instruction _ lanes => (List.ofFn lanes).filterMap id

def Micro.eval (a : Ops I F H) (xs ys : Fin BK → BitVec 8) (seed : I) : Micro BK → I
  | .scalar k => a.iadd seed (a.product8 (xs k) (ys k))
  | .instruction kind lanes => a.mma32 kind
      (fun i => ((lanes i).map xs).getD 0) (fun i => ((lanes i).map ys).getD 0) seed

/-- This is selected at a particular source dot+add site. Coverage is a
permutation of exactly its BK operands; holes can only be explicit zero lanes.
It is a structural compiler obligation, independent of all numeric values. -/
structure DotPlan (BK : Nat) where
  operations : List (Micro BK)
  complete : (operations.flatMap Micro.terms).Perm (List.finRange BK)
  fused : Bool

def DotPlan.eval (p : DotPlan BK) (a : Ops I F H) (xs ys : Fin BK → BitVec 8) (old : I) : I :=
  if p.fused then p.operations.foldl (fun acc op => op.eval a xs ys acc) old
  else a.iadd old (p.operations.foldl (fun acc op => op.eval a xs ys acc) a.izero)

/-- Provenance identifies the actual source operand, dot, and old-accumulator
IDs. A fused plan must consume that same old accumulator; there is no field
allowing a later output value to be substituted for it. -/
structure DotOrigin where
  site : Nat
  leftID : Nat
  rightID : Nat
  dotID : Nat
  oldID : Nat
  resultID : Nat
  leftEarlier : leftID<dotID
  rightEarlier : rightID<dotID
  dotEarlier : dotID<resultID
  oldEarlier : oldID<resultID

structure Selection (s : Shape) (c : Config) where
  plan : Nat → Nat → Nat → Nat → DotPlan c.BK
  origin : Nat → Nat → Nat → Nat → DotOrigin

abbrev Address := Nat × Nat
abbrev Memory (F H : Type) := Address → Sum (BitVec 8) (Sum F H)

structure Buffers where
  a : Nat
  b : Nat
  x : Nat
  w : Nat
  bias : Nat
  out : Nat
  aOffset : Nat
  bOffset : Nat
  xOffset : Nat
  wOffset : Nat
  biasOffset : Nat
  outOffset : Nat

def Buffers.Fresh (b : Buffers) : Prop :=
  b.a≠b.out ∧ b.b≠b.out ∧ b.x≠b.out ∧ b.w≠b.out ∧ b.bias≠b.out
def read8 (m : Memory F H) (p : Address) : BitVec 8 :=
  match m p with | .inl x => x | _ => 0
def read32 (a : Ops I F H) (m : Memory F H) (p : Address) : F :=
  match m p with | .inr (.inl x) => x | _ => a.fzero
def read16 (a : Ops I F H) (m : Memory F H) (p : Address) : H :=
  match m p with | .inr (.inr x) => x | _ => a.hzero
def write16 (m : Memory F H) (p : Address) (x : H) : Memory F H :=
  fun q => if q=p then .inr (.inr x) else m q

def operandA (s : Shape) (c : Config) (b : Buffers) (m : Memory F H)
    (r t z : Nat) (l : Fin c.BK) : BitVec 8 :=
  if sourceK c t z l.val<s.K then read8 m (b.a,b.aOffset+r*s.K+sourceK c t z l.val) else 0
def operandB (s : Shape) (c : Config) (b : Buffers) (m : Memory F H)
    (n t z : Nat) (l : Fin c.BK) : BitVec 8 :=
  if sourceK c t z l.val<s.K then read8 m (b.b,b.bOffset+sourceK c t z l.val*s.N+n) else 0

/-- This recursion is the source loop; in particular every selected dot plan
gets the actual preceding loop accumulator. -/
def accumulation (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (m : Memory F H) (r n z : Nat) : Nat → I
  | 0 => a.izero
  | t+1 => (selected.plan r n z t).eval a (operandA s c b m r t z) (operandB s c b m n t z)
      (accumulation a s c b selected m r n z t)

/-- Runtime divisor is an FP32 ABI word, not an exact Real source literal.
The order is int32→FP32, mul divisor, mul row scale, mul column scale, FP16
narrowing, then (split 0 only) FP32 bias→FP16 and FP16 add. -/
def partialValue (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (divisor : F) (hasBias : Bool) (m : Memory F H) (r n z : Nat) : H :=
  let acc := accumulation a s c b selected m r n z (steps s c)
  let x := read32 a m (b.x,b.xOffset+r)
  let w := read32 a m (b.w,b.wOffset+n)
  let value := a.narrow (a.mul w (a.mul x (a.mul (a.toFloat acc) divisor)))
  if hasBias && z==0 then a.hadd value (a.narrow (read32 a m (b.bias,b.biasOffset+n))) else value

def InputsAgree (b : Buffers) (m n : Memory F H) : Prop :=
  ∀ p,p.1≠b.out → m p=n p
theorem accumulation_current (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (fresh : b.Fresh) (selected : Selection s c) (m m' : Memory F H)
    (same : InputsAgree b m m') (r n z t : Nat) :
    accumulation a s c b selected m r n z t=accumulation a s c b selected m' r n z t := by
  have ha : operandA s c b m r=operandA s c b m' r := by
    funext t z l
    simp only [operandA,read8]
    rw [same _ fresh.1]
  have hb : operandB s c b m n=operandB s c b m' n := by
    funext t z l
    simp only [operandB,read8]
    rw [same _ fresh.2.1]
  induction t with
  | zero => rfl
  | succ t ih => simp only [accumulation,ha,hb,ih]
theorem partial_current (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (fresh : b.Fresh) (selected : Selection s c) (divisor : F) (hasBias : Bool)
    (m m' : Memory F H) (same : InputsAgree b m m') (r n z : Nat) :
    partialValue a s c b selected divisor hasBias m r n z=partialValue a s c b selected divisor hasBias m' r n z := by
  unfold partialValue read32
  rw [accumulation_current a s c b fresh selected m m' same,
    same _ fresh.2.2.1,same _ fresh.2.2.2.1,same _ fresh.2.2.2.2]

abbrev Word32 := FP.Value .fp32
abbrev Word16 := TritonBenchG.MatmulDequantInt4.Binary16.Value

def narrow16 (x : Word32) : Word16 := TritonBenchG.QuantizeKvTransform.MixedBits.half (TritonBenchG.QuantizeKvTransform.MixedBits.narrow .ieeeValue x)
def add16 := TritonBenchG.MatmulDequantInt4.Binary16.add .ieeeValue

/-- Each port is precisely one fixed instruction's one output lane. The
selected kind records saturating MMA versus non-saturating WGMMA. It contains
no variable K, scales, bias, output matrix, or final-memory argument. -/
structure FixedISAProfile where
  result : ISA → (Fin 32 → BitVec 8) → (Fin 32 → BitVec 8) → BitVec 32 → BitVec 32

def fpOps (profile : FixedISAProfile) : Ops (BitVec 32) Word32 Word16 where
  izero := 0
  fzero := FP.Value.zero .fp32
  hzero := TritonBenchG.MatmulDequantInt4.Binary16.Value.ofBits 0
  product8 := fun x y => BitVec.ofInt 32 (x.toInt*y.toInt)
  iadd := (·+·)
  mma32 := profile.result
  toFloat := fun x => FP.round .fp32 .ieeeValue (Rat.ofInt x.toInt)
  mul := FP.mul .ieeeValue
  narrow := narrow16
  hadd := add16

structure PrimitiveLeaves (a : Ops (BitVec 32) Word32 Word16) (profile : FixedISAProfile) : Prop where
  izero : a.izero=0
  fzero : a.fzero=FP.Value.zero .fp32
  hzero : a.hzero=TritonBenchG.MatmulDequantInt4.Binary16.Value.ofBits 0
  product : ∀ x y,a.product8 x y=BitVec.ofInt 32 (x.toInt*y.toInt)
  iadd : ∀ x y,a.iadd x y=x+y
  instruction : ∀ kind xs ys old,a.mma32 kind xs ys old=profile.result kind xs ys old
  convert : ∀ x,a.toFloat x=FP.round .fp32 .ieeeValue (Rat.ofInt x.toInt)
  mul : ∀ x y,a.mul x y=FP.mul .ieeeValue x y
  narrow : ∀ x,a.narrow x=narrow16 x
  hadd : ∀ x y,a.hadd x y=add16 x y

theorem PrimitiveLeaves.eq (h : PrimitiveLeaves a profile) : a=fpOps profile := by
  cases a
  simp only [fpOps,Ops.mk.injEq]
  exact ⟨h.izero,h.fzero,h.hzero,funext (fun x => funext (h.product x)),
    funext (fun x => funext (h.iadd x)),
    funext (fun kind => funext (fun xs => funext (fun ys => funext (h.instruction kind xs ys)))),
    funext h.convert,funext (fun x => funext (h.mul x)),funext h.narrow,
    funext (fun x => funext (h.hadd x))⟩

#axiomsClean accumulation_current
#axiomsClean partial_current
#axiomsClean PrimitiveLeaves.eq
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
