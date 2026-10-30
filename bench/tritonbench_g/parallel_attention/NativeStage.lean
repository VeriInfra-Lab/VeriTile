import bench.tritonbench_g.parallel_attention.ScalarModel
import bench.tritonbench_g.parallel_attention.Proofs.NativePlans

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 3000000

abbrev Word := BitVec 32
inductive Phase where | forward | query | keyValue deriving DecidableEq
inductive DotRole where | score | cotangent | output | coefficient deriving DecidableEq

/-- Staged source expressions retain a dot's scalar operands and live seed.
Input words never select compiler routes; only producer syntax and addresses do. -/
inductive Expr where
  | input (region : RegionName) (offset : Nat) (word : Word)
  | literal (value : ℝ)
  | add (x y : Expr)
  | mul (x y : Expr)
  | cast (x : Expr)
  | dot (role : DotRole) (start : Option Nat) (row col k : Nat)
      (left right : Fin k → Expr) (seed : Expr) (pending : Bool)
  | scaledSum (dot z scale : Expr)
  | normalizer (n : Nat) (inputs : Fin n → Expr)
instance : Inhabited Expr := ⟨.literal 0⟩

/-- Producer metadata only. Literal or FP payloads are not inspected. -/
def origin : Expr → Option (RegionName×Nat)
  | .input R off _ => some (R,off)
  | .cast x => origin x
  | .mul x y | .add x y => (origin x).or (origin y)
  | _ => none

def coefficient : Expr → Bool
  | .cast x => coefficient x
  | .mul (.mul (.literal _) _) _ => Bool.true
  | _ => Bool.false

def role (phase : Phase) (left : Expr) : DotRole :=
  if coefficient left then .coefficient else
  match origin left with
  | some ("q",_) | some ("k",_) => .score
  | some ("v",_) | some ("do",_) => .cotangent
  | _ => if phase=.forward then .output else .output

def timeOf (shape : Shape) : Expr → Option Nat
  | .input R off _ =>
      if R="q" ∨ R="k" then some ((off%(shape.T*shape.K))/shape.K)
      else if R="v" ∨ R="do" then some ((off%(shape.T*shape.V))/shape.V)
      else none
  | .cast x => timeOf shape x
  | _ => none

/-- Source dot production is fixed; all source dots use allow_tf32=Bool.false and
FP32 operands. The scalar coordinate order is retained for the local lowering. -/
def dot (phase : Phase) (shape : Shape) : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Expr) → (TileIndex (batch++[k,n]) → Expr) →
    TileIndex (batch++[m,n]) → Expr
  | [],_,k,_,a,b,i =>
      let left := fun f => a ⟨i.1,f,()⟩
      let right := fun f => b ⟨f,i.2.1,()⟩
      let r := if h : 0<k then role phase (left ⟨0,h⟩) else .output
      let t := if h : 0<k then (timeOf shape (right ⟨0,h⟩)).map
        (fun t => if r=.score ∨ r=.cotangent then t-i.2.1.val else t) else none
      .dot r t i.1.val i.2.1.val k left right (.literal 0) Bool.true
  | _::rest,_,_,_,a,b,i => dot phase shape rest (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def dots (phase : Phase) (shape : Shape) : DotPrimitive Expr where
  apply := fun _ _ {batch} {_} {_} {_} a b => dot phase shape batch a b

/-- Local source contractions. A pending dot remains available until its
consumer supplies the live accumulator; no rounded add is equated with FMA. -/
local instance : DecidableEq ℝ := Classical.decEq _

def consume (phase : Phase) (d : NativePlans.Descriptor) : Binary → Expr → Expr → Expr
  | .add,acc,.dot r t row col k a b _ Bool.true => .dot r t row col k a b acc Bool.false
  | .add,.dot r t row col k a b z Bool.true,acc =>
      if d.postSeed then
        if phase=.keyValue then .add acc (.dot r t row col k a b z Bool.false)
        else .add (.dot r t row col k a b z Bool.false) acc
      else .dot r t row col k a b acc Bool.false
  | .add,.mul x@(.dot ..) scale,.mul z _ => .scaledSum x z scale
  | .add,x,y => .add x y
  | .mul,x,.literal scale => if scale=1 ∧ d.shape.useScale=Bool.false then x else .mul x (.literal scale)
  | .mul,x,y => .mul x y
  | _,_,_ => .literal 0

def reduce {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : TileIndex shape → Expr) : TileIndex (TileShape.reduceShape shape ax keep) → Expr :=
  match keep with
  | Bool.false => fun out => .normalizer (TileShape.axisDim shape ax)
      (fun j => xs (TileShape.insertAxisIndex shape ax out j))
  | Bool.true => fun out => .normalizer (TileShape.axisDim shape ax)
      (fun j => xs (TileShape.replaceAxisIndex shape ax out j))

def algebra (phase : Phase) (d : NativePlans.Descriptor) : Algebra Expr where
  literal := fun _ _ => Expr.literal
  negInf := .literal 0
  binary := fun _ _ => consume phase d
  unary := fun _ _ _ => .literal 0
  cast := fun _ _ _ => Expr.cast
  fromNat := fun _ n => .literal n
  fromInt := fun _ n => .literal n
  fp32Bits := fun w => .input "literalBits" 0 w.bits
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => .literal 0
  reduceSum := fun _ {_} ax keep xs => reduce ax keep xs

/-- Fixed compiler descriptor with fixed task-local source plans. There is no
caller-selected reduction tree, host sum, numerical callback or output value. -/
def layout (d : NativePlans.Descriptor) : CompilerLayout := NativePlans.layout d sourcePlans

/-- The normalizer's source square remains available to the local FMA tree. -/
def squareRoot : Expr → Expr
  | .mul x _ => x
  | .cast x => squareRoot x
  | x => x

def dotStart : Expr → Option Nat
  | .dot _ t .. => t
  | .cast x => dotStart x
  | .mul x y | .add x y => (dotStart x).or (dotStart y)
  | .scaledSum x _ _ => dotStart x
  | _ => none

def diagonal (phase : Phase) (tile : Tile) (start : Option Nat) : Bool :=
  match start with
  | none => Bool.true
  | some n => if phase=.forward then decide (tile.chunk*128≤n)
      else if phase=.query then decide (tile.chunk*64≤n)
      else decide (n<(tile.chunk+1)*64)

def iteration (phase : Phase) (shape : Shape) (tile : Tile) (diag : Bool) (start : Nat) : Nat :=
  if diag then (start-tile.chunk*(if phase=.forward then 128 else 64))/32
  else if phase=.keyValue then blocks shape.T 32-1-start/32 else start/32

def coefficientWord (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor) (tile : Tile)
    (diag valid : Bool) (row j u : Nat) (ds scale score : Word) : Word :=
  let site := [3,if phase=.query then 9 else 10,tile.bh,tile.chunk,tile.value,row,j,u,diag.toNat]
  let scalar := (layout d).scalar
  let twice := fun value => if diag && d.shape.useScale then
    a.fma scale value (a.mul scale value) else a.add value value
  let product := (scalar.coefficientOrder site).eval (a.ops 0)
  match scalar.coefficientMask site with
  | .beforeTwice => product (twice (if valid then ds else a.literal 0))
      (if valid then score else a.literal 0)
  | .afterTwice => product (if valid then twice ds else a.literal 0)
      (if valid then score else a.literal 0)
  | .afterProduct => if valid then product (twice ds) score else a.literal 0

/- Native word evaluation has fixed-arity literal/add/mul/FMA ports only.
Each coefficient is consumed by its parent dot with the source row/column
coordinates, so mask placement survives even when a selected branch is zero. -/
mutual
 def evaluate (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor) (tile : Tile) (e : Expr) : Word := match e with
  | .input _ _ word => word
  | .literal x => a.literal x
  | .cast x => evaluate a phase d tile x
  | .add x y => a.add (evaluate a phase d tile x) (evaluate a phase d tile y)
  | .mul x (.literal scale) =>
      a.mul (a.literal scale) (evaluate a phase d tile x)
  | .mul x y => a.mul (evaluate a phase d tile x) (evaluate a phase d tile y)
  | .scaledSum x z scale =>
      if d.scaledSumOnZ then a.fma (evaluate a phase d tile scale) (evaluate a phase d tile z)
        (a.mul (evaluate a phase d tile scale) (evaluate a phase d tile x))
      else a.fma (evaluate a phase d tile scale) (evaluate a phase d tile x)
        (a.mul (evaluate a phase d tile scale) (evaluate a phase d tile z))
  | .dot dotRole start row _ k left right seed _ =>
      let diag := diagonal phase tile start
      let begin := start.getD ((tile.chunk+1)*64)
      let j := iteration phase d.shape tile diag begin
      (DotPlan.linear k).eval (a.ops 0)
        (fun u => if h : u<k then
          if dotRole=.coefficient then
            let valid := if phase=.query then !diag || decide (j*32+u≤row)
              else decide (begin+u<d.shape.T) && (!diag || decide (row≤j*32+u))
            evaluateCoefficient a phase d tile diag valid begin row j u (left ⟨u,h⟩)
          else evaluate a phase d tile (left ⟨u,h⟩) else a.literal 0)
        (fun j => if h : j<k then evaluate a phase d tile (right ⟨j,h⟩) else a.literal 0)
        (evaluate a phase d tile seed)
  | .normalizer n xs =>
      let first := if h : 0<n then xs ⟨0,h⟩ else .literal 0
      let diag := diagonal phase tile (dotStart first)
      let raw := fun j => if h : j<n then evaluateSquareRoot a phase d tile (xs ⟨j,h⟩) else a.literal 0
      let squared := fun j => if h : j<n then evaluate a phase d tile (xs ⟨j,h⟩) else a.literal 0
      let rec norm : DotPlan → Word
        | .zero | .acc => a.literal 0
        | .product j => squared j
        | .add l r => a.add (norm l) (norm r)
        | .fma j rest => a.fma (raw j) (raw j) (norm rest)
      norm ((layout d).normTree [diag.toNat] n)
 termination_by structural e

 def evaluateSquareRoot (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor) (tile : Tile) (e : Expr) : Word := match e with
  | .mul x _ => evaluate a phase d tile x
  | .cast x => evaluateSquareRoot a phase d tile x
  | .literal x => a.literal x
  | _ => a.literal 0
 termination_by structural e

 def evaluateCoefficient (a : Scalars Word) (phase : Phase) (d : NativePlans.Descriptor) (tile : Tile)
    (diag valid : Bool) (begin row j u : Nat) (e : Expr) : Word := match e with
  | .cast x => evaluateCoefficient a phase d tile diag valid begin row j u x
  | .mul (.mul (.literal _) (.mul raw (.literal scale))) score =>
      let value := evaluate a phase d tile raw
      coefficientWord a phase d tile diag valid row j u
        (if diag then value else a.mul (a.literal scale) value) (a.literal scale)
        (evaluate a phase d tile score)
  | .mul (.mul (.literal _) ds) score =>
      coefficientWord a phase d tile diag valid row j u
        (evaluate a phase d tile ds) (a.literal 1) (evaluate a phase d tile score)
  | _ => a.literal 0
 termination_by structural e
end

/-- Scalar instruction agreement only; no dot, loop, output, memory or
source/native whole-result hypothesis is part of the primitive boundary. -/
structure PrimitiveLaws (actual reference : Scalars Word) : Prop where
  literal : ∀ x,actual.literal x=reference.literal x
  add : ∀ x y,actual.add x y=reference.add x y
  mul : ∀ x y,actual.mul x y=reference.mul x y
  fma : ∀ x y z,actual.fma x y z=reference.fma x y z
  cast : ∀ x,actual.cast x=reference.cast x
  div : ∀ x y,actual.div x y=reference.div x y
  neg : ∀ x,actual.neg x=reference.neg x

theorem PrimitiveLaws.eq {actual reference : Scalars Word} (h : PrimitiveLaws actual reference) : actual=reference := by
  cases actual; cases reference
  rcases h with ⟨hl,ha,hm,hf,hc,hd,hn⟩
  simp only [Scalars.mk.injEq]
  exact ⟨funext hl,funext (fun x => funext (ha x)),funext (fun x => funext (hm x)),
    funext (fun x => funext (fun y => funext (hf x y))),funext hc,
    funext (fun x => funext (hd x)),funext hn⟩

#axiomsClean PrimitiveLaws.eq
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
