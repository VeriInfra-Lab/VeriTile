import bench.tritonbench_g.matmul_dequantize.Proofs.NativeInstructions
import bench.tritonbench_g.matmul_dequantize.Proofs.SecondComplete
import bench.tritonbench_g.matmul_dequantize.Proofs.Mathematical_FPFirstReference

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

inductive Edge where
  | left | right | seed | factorLeft | factorRight
  | dotLeft (i : Nat) | dotRight (i : Nat)
  deriving DecidableEq

inductive Expr where
  | bits16 (word : BitVec 16)
  | bits32 (word : BitVec 32)
  | literal (dtype : FloatDType) (value : ℝ)
  | integer (value : Int)
  | binary (dtype : FloatDType) (op : Binary) (a b : Expr)
  | cast (src dst : FloatDType) (value : Expr)
  | pendingDot (D row col : Nat) (a b : Fin D → Expr)

instance : Inhabited Expr := ⟨.bits32 0⟩

/-- Only compiler routing and actual multiply orientation; no numerical
function, tensor result, whole-dot value or execution proposition. -/
structure Plan where
  route : Expr → Edge → Nat → Nat
  reverseMul : Expr → Nat → Bool

def lower (plan : Plan) : Expr → Nat → Graph
  | .bits16 w,_ => .bits16 w
  | .bits32 w,_ => .bits32 w
  | .literal dtype x,_ => if dtype=.fp16 then .literal16 x else .literal32 x
  | .integer x,_ => .literal32 x
  | .cast _ .fp16 (.integer x),_ => .literal16 x
  | e@(.cast src dst x),owner =>
      let value := lower plan x (plan.route e .left owner)
      if src=dst then value
      else if dst=.fp16 then .narrow value
      else if src=.fp16 then .widen value else value
  | e@(.binary _ .add seed (.pendingDot D row col a b)),owner =>
      dot D row col
        (fun i => lower plan (a i) (plan.route e (.dotLeft i.val) owner))
        (fun i => lower plan (b i) (plan.route e (.dotRight i.val) owner))
        (lower plan seed (plan.route e .seed owner))
  | e@(.binary .fp16 .sub (.binary .fp16 .mul a b) c),owner =>
      let av := lower plan a (plan.route e .factorLeft owner)
      let bv := lower plan b (plan.route e .factorRight owner)
      let cv := Graph.neg16 (lower plan c (plan.route e .seed owner))
      if plan.reverseMul (.binary .fp16 .mul a b) owner then .fma16 bv av cv else .fma16 av bv cv
  | e@(.binary dtype op a b),owner =>
      let av := lower plan a (plan.route e .left owner)
      let bv := lower plan b (plan.route e .right owner)
      match op with
      | .add => .add32 av bv
      | .mul =>
          if dtype=.fp16 then
            if plan.reverseMul e owner then .mul16 bv av else .mul16 av bv
          else if plan.reverseMul e owner then .mul32 bv av else .mul32 av bv
      | .sub => .sub16 av bv
      | _ => .bits32 0
  | e@(.pendingDot D row col a b),owner =>
      dot D row col
        (fun i => lower plan (a i) (plan.route e (.dotLeft i.val) owner))
        (fun i => lower plan (b i) (plan.route e (.dotRight i.val) owner)) (.bits32 0)

/-- Explicit bit-preserving casts keep a pending MMA live as the add consumer;
all half narrowing and widening remain represented. -/
def castExpr (src dst : FloatDType) (x : Expr) : Expr :=
  if src=dst ∨ ((src=.real ∨ src=.fp32) ∧ (dst=.real ∨ dst=.fp32)) then x else .cast src dst x

def scalar : Algebra Expr where
  literal := fun _ dtype x => .literal dtype x
  negInf := .bits32 0xFF800000
  binary := fun _ dtype op a b => .binary dtype op a b
  unary := fun _ _ x => x
  cast := fun _ src dst x => castExpr src dst x
  fromNat := fun _ n => .integer n
  fromInt := fun _ n => .integer n
  fp32Bits := fun b => .bits32 b.bits
  fp32Load := id
  reduceSum := fun _ _ _ _ _ => default
  reduceMax := fun _ _ _ _ _ => default

def dotValues : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Expr) → (TileIndex (batch++[k,n]) → Expr) →
    (TileIndex (batch++[m,n]) → Expr)
  | [],_,k,_,a,b,i => .pendingDot k i.1.val i.2.1.val
      (fun d => a ⟨i.1,d,()⟩) (fun d => b ⟨d,i.2.1,()⟩)
  | _::ds,_,_,_,a,b,i => dotValues ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def firstDot : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Dot Expr where
  apply := fun _ _ {batch} {_} {_} {_} a b => dotValues batch a b

def secondDot : FPTypedSecond.Dot Expr where
  apply := fun _ _ {batch} {_} {_} {_} a b => dotValues batch a b

theorem preserves_default_half_fma (p : Plan) (a b c : Expr) (owner : Nat) :
    lower p (.binary .fp16 .sub (.binary .fp16 .mul a b) c) owner=
      (let e := Expr.binary .fp16 .sub (.binary .fp16 .mul a b) c
       let av := lower p a (p.route e .factorLeft owner)
       let bv := lower p b (p.route e .factorRight owner)
       let cv := Graph.neg16 (lower p c (p.route e .seed owner))
       if p.reverseMul (.binary .fp16 .mul a b) owner then .fma16 bv av cv else .fma16 av bv cv) := rfl

theorem preserves_live_seed (p : Plan) (a b : Fin D → Expr) (seed : Expr) (row col owner : Nat) :
    lower p (.binary .real .add seed (.pendingDot D row col a b)) owner=
      (let e := Expr.binary .real .add seed (.pendingDot D row col a b)
       dot D row col
         (fun i => lower p (a i) (p.route e (.dotLeft i.val) owner))
         (fun i => lower p (b i) (p.route e (.dotRight i.val) owner))
         (lower p seed (p.route e .seed owner))) := rfl

#axiomsClean preserves_default_half_fma
#axiomsClean preserves_live_seed
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Native
