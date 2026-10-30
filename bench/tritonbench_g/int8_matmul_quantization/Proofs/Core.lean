import bench.tritonbench_g.int8_dequant_matmul.Proofs.Plan

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000

abbrev Shape := Int8DequantMatmul.Shape
abbrev MatrixCell := Int8DequantMatmul.MatrixCell
abbrev Address := Int8DequantMatmul.Address
abbrev Memory := Int8DequantMatmul.Memory
abbrev ISA := Int8DequantMatmul.ISA
abbrev Micro := Int8DequantMatmul.Micro
abbrev DotPlan := Int8DequantMatmul.DotPlan
abbrev Byte := BitVec 8
abbrev Word32 := Int8DequantMatmul.Word32
abbrev Word16 := Int8DequantMatmul.Word16

inductive Precision where
  | half | single
  deriving DecidableEq

/-- Fixed scalar operations only. DotPlan contains a proved finite structural
cover and its fixed K32 instructions consume the actual old accumulator. -/
structure Ops (I F H : Type) extends Int8DequantMatmul.Ops I F H where
  widen : H → F
  add : F → F → F
  absolute : F → F
  maximum : F → F → F
  divide : F → F → F
  scale127 : F → F
  integer8 : F → Byte

abbrev read8 := @Int8DequantMatmul.read8
abbrev read32 := @Int8DequantMatmul.read32
abbrev read16 := @Int8DequantMatmul.read16
abbrev write16 := @Int8DequantMatmul.write16
def write8 (m : Memory F H) (p : Address) (x : Byte) : Memory F H :=
  fun q => if q=p then .inl x else m q
def write32 (m : Memory F H) (p : Address) (x : F) : Memory F H :=
  fun q => if q=p then .inr (.inl x) else m q

abbrev Stored (F H : Type) := Sum F H
def readStored (a : Ops I F H) (precision : Precision) (m : Memory F H) (p : Address) : Stored F H :=
  match m p with
  | .inr value => value
  | .inl _ => match precision with | .half => .inr a.hzero | .single => .inl a.fzero
def writeStored (m : Memory F H) (p : Address) (x : Stored F H) : Memory F H :=
  fun q => if q=p then .inr x else m q
def loadScale (a : Ops I F H) (precision : Precision) (m : Memory F H) (p : Address) : F :=
  match precision with
  | .half => a.widen (read16 a.toOps m p)
  | .single => read32 a.toOps m p
def outputZero (a : Ops I F H) : Precision → Stored F H
  | .half => .inr a.hzero
  | .single => .inl a.fzero
def contribution (a : Ops I F H) : Precision → H → Stored F H
  | .half,x => .inr x
  | .single,x => .inl (a.widen x)
def outputAdd (a : Ops I F H) (precision : Precision) (old : Stored F H) (value : H) : Stored F H :=
  match precision with
  | .half => .inr (a.hadd (match old with | .inr x => x | _ => a.hzero) value)
  | .single => .inl (a.add (match old with | .inl x => x | _ => a.fzero) (a.widen value))

structure Config extends Int8DequantMatmul.Config where
  group : Nat
  groupPositive : 0<group
def steps (s : Shape) (c : Config) := Int8DequantMatmul.steps s c.toConfig
def sourceK (c : Config) (t z l : Nat) := Int8DequantMatmul.sourceK c.toConfig t z l
abbrev Selection (s : Shape) (c : Config) := Int8DequantMatmul.Selection s c.toConfig

/-- Canonical row-major A and row/column-major B include the actual producer
outputs; offsets are element offsets and may be nonzero. -/
structure Buffers where
  a : Nat
  b : Nat
  aScale : Nat
  bScale : Nat
  out : Nat
  aOffset : Nat
  bOffset : Nat
  aScaleOffset : Nat
  bScaleOffset : Nat
  outOffset : Nat
  bColumnMajor : Bool
  aScalePrecision : Precision
  bScalePrecision : Precision
  outPrecision : Precision
def Buffers.Separate (b : Buffers) : Prop :=
  b.a≠b.out ∧ b.b≠b.out ∧ b.aScale≠b.out ∧ b.bScale≠b.out
def Buffers.bIndex (b : Buffers) (s : Shape) (k n : Nat) : Nat :=
  b.bOffset+if b.bColumnMajor then n*s.K+k else k*s.N+n
def outputAddress (s : Shape) (b : Buffers) (x : MatrixCell s) : Address :=
  (b.out,b.outOffset+x.1.val*s.N+x.2.val)
def InputsAgree (b : Buffers) (m n : Memory F H) : Prop :=
  ∀ p,p.1≠b.out → m p=n p

theorem outputAddress_injective (s : Shape) (b : Buffers) : Function.Injective (outputAddress s b) := by
  intro x y same
  have h := congrArg Prod.snd same
  simp only [outputAddress,Prod.snd] at h
  have hx := x.2.isLt
  have hy := y.2.isLt
  have hn : 0<s.N := by omega
  have eq : x.1.val*s.N+x.2.val=y.1.val*s.N+y.2.val := by omega
  have divs := congrArg (fun k => k/s.N) eq
  simp only [Nat.mul_comm _ s.N,Nat.mul_add_div hn,Nat.div_eq_of_lt hx,Nat.div_eq_of_lt hy,Nat.add_zero] at divs
  apply Prod.ext
  · exact Fin.ext divs
  · apply Fin.ext
    rw [divs] at eq
    omega

def operandA (s : Shape) (c : Config) (b : Buffers) (m : Memory F H)
    (r t z : Nat) (l : Fin c.BK) : Byte :=
  if sourceK c t z l.val<s.K then read8 m (b.a,b.aOffset+r*s.K+sourceK c t z l.val) else 0
def operandB (s : Shape) (c : Config) (b : Buffers) (m : Memory F H)
    (n t z : Nat) (l : Fin c.BK) : Byte :=
  if sourceK c t z l.val<s.K then read8 m (b.b,b.bIndex s (sourceK c t z l.val) n) else 0
def accumulation (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (m : Memory F H) (r n z : Nat) : Nat → I
  | 0 => a.izero
  | t+1 => (selected.plan r n z t).eval a.toOps (operandA s c b m r t z) (operandB s c b m n t z)
      (accumulation a s c b selected m r n z t)

/-- Original order and mandatory half narrowing: signed32→FP32, row scale,
column scale, then FP16. Destination widening/atomic arithmetic occurs later. -/
def partialValue (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (selected : Selection s c) (m : Memory F H) (r n z : Nat) : H :=
  let acc := accumulation a s c b selected m r n z (steps s c)
  let sa := loadScale a b.aScalePrecision m (b.aScale,b.aScaleOffset+r)
  let sb := loadScale a b.bScalePrecision m (b.bScale,b.bScaleOffset+n)
  a.narrow (a.mul (a.mul (a.toFloat acc) sa) sb)

theorem accumulation_current (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (sep : b.Separate) (selected : Selection s c) (m m' : Memory F H)
    (same : InputsAgree b m m') (r n z t : Nat) :
    accumulation a s c b selected m r n z t=accumulation a s c b selected m' r n z t := by
  have ha : operandA s c b m r=operandA s c b m' r := by
    funext t z l
    simp only [operandA,Int8DequantMatmul.read8]
    rw [same _ sep.1]
  have hb : operandB s c b m n=operandB s c b m' n := by
    funext t z l
    simp only [operandB,Int8DequantMatmul.read8]
    rw [same _ sep.2.1]
  induction t with
  | zero => rfl
  | succ t ih => simp only [accumulation,ha,hb,ih]

theorem partial_current (a : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (sep : b.Separate) (selected : Selection s c) (m m' : Memory F H)
    (same : InputsAgree b m m') (r n z : Nat) :
    partialValue a s c b selected m r n z=partialValue a s c b selected m' r n z := by
  unfold partialValue loadScale read32 read16 Int8DequantMatmul.read32 Int8DequantMatmul.read16
  rw [accumulation_current a s c b sep selected m m' same,
    same _ sep.2.2.1,same _ sep.2.2.2]

#axiomsClean outputAddress_injective
#axiomsClean accumulation_current
#axiomsClean partial_current
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
