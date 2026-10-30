import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
namespace Native

inductive Format where
  | half | bfloat | single
  deriving DecidableEq, Repr
inductive WeightFormat where
  | half | bfloat
  deriving DecidableEq, Repr

def WeightFormat.format : WeightFormat → Format
  | .half => .half | .bfloat => .bfloat

def Format.bits : Format → Nat
  | .half | .bfloat => 16 | .single => 32
abbrev Word (f : Format) := BitVec f.bits
abbrev Matrix (f : Format) (rows columns : Nat) := Fin rows → Fin columns → Word f

/-- Exact local instruction ports. The MMA port has the native m16n8k16
low-input/FP32-accumulator shape, and receives the incoming accumulator.
There is no whole-output, successful-execution or memory/index callback. -/
structure Primitives where
  convert : (source target : Format) → Word source → Word target
  add : (format : Format) → Word format → Word format → Word format
  mma : (format : WeightFormat) → Matrix format.format 16 16 →
    Matrix format.format 16 8 → Matrix .single 16 8 → Matrix .single 16 8

/-- The structural interface is inhabited without any numerical-law premise.
This witness is not proposed as hardware arithmetic. -/
def zeroPrimitives : Primitives where
  convert := fun _ _ _ => 0
  add := fun _ _ _ => 0
  mma := fun _ _ _ _ _ _ => 0

/-- A no-op conversion is an actual unchanged raw word, including NaN payloads. -/
def convertIfNeeded (ops : Primitives) (source target : Format) (x : Word source) : Word target :=
  if h : source = target then h ▸ x
  else if source = .single ∨ target = .single then ops.convert source target x
  else ops.convert .single target (ops.convert source .single x)

def fragmentRow (r : Fin 32) (i : Fin 16) : Fin 32 :=
  ⟨r.val / 16 * 16 + i.val, by have hr := r.isLt; have hi := i.isLt; omega⟩
def fragmentColumn (c : Fin 32) (j : Fin 8) : Fin 32 :=
  ⟨c.val / 8 * 8 + j.val, by have hc := c.isLt; have hj := j.isLt; omega⟩

/-- Eight disjoint m16n8k16 fragments form the source's 32×32×16 dotAcc.
No sequential scalar FMA or independent dot-plus-add substitutes for MMA. -/
def mmaTile (ops : Primitives) (f : WeightFormat)
    (a : Matrix f.format 32 16) (b : Matrix f.format 16 32)
    (acc : Matrix .single 32 32) : Matrix .single 32 32 := fun r c =>
  ops.mma f
    (fun i k => a (fragmentRow r i) k)
    (fun k j => b k (fragmentColumn c j))
    (fun i j => acc (fragmentRow r i) (fragmentColumn c j))
    ⟨r.val % 16,Nat.mod_lt _ (by decide)⟩
    ⟨c.val % 8,Nat.mod_lt _ (by decide)⟩

structure Profile where
  input : Format
  weight : WeightFormat
  output : Format
  addInputs : Bool
  deriving DecidableEq

def defaultHalfProfile : Profile := ⟨.single,.half,.single,Bool.false⟩
def defaultBfloatProfile : Profile := ⟨.single,.bfloat,.single,Bool.false⟩

/-- Triton's mixed half/bfloat addition selects half; FP32 output selects FP32. -/
def additionFormat (p : Profile) : Format := match p.output,p.weight with
  | .single,_ => .single
  | .half,_ => .half
  | .bfloat,.half => .half
  | .bfloat,.bfloat => .bfloat

/-- Bounded SM90 lowering distinguishes mixed FP16-weight/BF16-output add:
that profile keeps computed-first; the other inspected profiles use old-first.
This table requires the ordinary local compiler contract. It is not an
assumption that floating addition commutes on raw NaN words. -/
def additionOldFirst (p : Profile) : Bool :=
  !(p.weight == .half && p.output == .bfloat)

def finish (ops : Primitives) (p : Profile) (acc : Word .single)
    (old : Word p.output) : Word p.output :=
  let rounded := ops.convert .single p.weight.format acc
  if p.addInputs then
    let format := additionFormat p
    let previous := convertIfNeeded ops p.output format old
    let computed := convertIfNeeded ops p.weight.format format rounded
    let added := if additionOldFirst p then ops.add format previous computed
      else ops.add format computed previous
    convertIfNeeded ops format p.output added
  else convertIfNeeded ops p.weight.format p.output rounded

/-- Entry logical snapshots; original caller aliases cannot affect subsequent
loads after the six private preparations succeed. -/
structure Inputs (p : Profile) where
  x : Nat → Nat → Word p.input
  weight : Nat → Nat → Nat → Word p.weight.format

/-- Padded input zeros are first words of the actual input format, then pass
through the source CAST_TYPE branch. No zero-preservation law is presumed. -/
def aTile (ops : Primitives) (p : Profile) (shape : Shape) (s : Segment)
    (input : Inputs p) (tileRow block : Nat) : Matrix p.weight.format 32 16 := fun r k =>
  let row := s.start + (tileRow*32+r.val) % s.length
  let inner := block*16+k.val
  let word : Word p.input := if inner < shape.inner then input.x row inner else 0
  convertIfNeeded ops p.input p.weight.format word

def bTile (p : Profile) (shape : Shape) (s : Segment) (input : Inputs p)
    (tileColumn block : Nat) : Matrix p.weight.format 16 32 := fun k c =>
  let column := (tileColumn*32+c.val) % shape.columns
  let inner := block*16+k.val
  if inner < shape.inner then input.weight (s.adapter.getD 0) column inner else 0

def accumulate (ops : Primitives) (p : Profile) (shape : Shape) (s : Segment)
    (input : Inputs p) (tileRow tileColumn : Nat) : Matrix .single 32 32 :=
  (List.range ((shape.inner+15)/16)).foldl
    (fun acc block => mmaTile ops p.weight
      (aTile ops p shape s input tileRow block)
      (bTile p shape s input tileColumn block) acc)
    (fun _ _ => 0)

def segmentStep (ops : Primitives) (p : Profile) (shape : Shape) (s : Segment)
    (input : Inputs p) (previous : Nat → Nat → Word p.output) (r c : Nat) : Word p.output :=
  if ActiveCell shape s r c then
    finish ops p
      (accumulate ops p shape s input ((r-s.start)/32) ((c-shape.slice)/32)
        ⟨(r-s.start)%32,Nat.mod_lt _ (by decide)⟩
        ⟨(c-shape.slice)%32,Nat.mod_lt _ (by decide)⟩)
      (previous r c)
  else previous r c

def result (ops : Primitives) (p : Profile) (shape : Shape) (segments : List Segment)
    (input : Inputs p) (entry : Nat → Nat → Word p.output) :=
  segments.foldl (fun y s => segmentStep ops p shape s input y) entry

/-- The only deferred numerical agreements are local conversion, selected-
format add and one m16n8k16 accumulator instruction, over all raw words. -/
structure PrimitiveAgreement (left right : Primitives) : Prop where
  convert : ∀ source target value, left.convert source target value = right.convert source target value
  add : ∀ format a b, left.add format a b = right.add format a b
  mma : ∀ format a b acc, left.mma format a b acc = right.mma format a b acc

theorem primitives_ext (left right : Primitives) (h : PrimitiveAgreement left right) : left = right := by
  cases left with | mk lc la lm =>
    cases right with | mk rc ra rm =>
      congr 1
      · funext s t x; exact h.convert s t x
      · funext f x y; exact h.add f x y
      · funext f a b acc; exact h.mma f a b acc

theorem result_of_local_primitives (left right : Primitives) (h : PrimitiveAgreement left right)
    (p : Profile) (shape : Shape) (segments : List Segment) (input : Inputs p)
    (entry : Nat → Nat → Word p.output) :
    result left p shape segments input entry = result right p shape segments input entry := by
  rw [primitives_ext left right h]

end Native
end VeriTile.Bench.Repairs.SgmvExpandSlice
