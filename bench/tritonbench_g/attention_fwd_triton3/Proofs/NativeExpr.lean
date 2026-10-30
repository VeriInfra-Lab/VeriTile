import bench.tritonbench_g.attention_fwd_triton3.Proofs.Reference
import bench.tritonbench_g.attention_fwd_triton3.Proofs.ScaleBinding

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile.Triton FP.Structural
noncomputable section

/-- A staged FP32 graph. The initial M role is retained as provenance for the
source alpha guard; its raw bits never choose a lowering or an owner route. -/
inductive Expr where
  | input (region : RegionName) (word : BitVec 32)
  | bits (word : BitVec 32)
  | literal (value : ℝ)
  | scale
  | negInf
  | initialCenter
  | center (previous blockMax : Expr)
  | add (a b : Expr)
  | product (a b : Expr)
  | difference (a b : Expr)
  | centerDifference (a b : Expr)
  | divide (a b : Expr)
  | exponential (a : Expr)
  | logarithm (a : Expr)
  | alpha (previous center : Expr)
  | fma (a b acc : Expr)
  | reduction (maximum : Bool) (row n : Nat) (values : Fin n → Expr)
  | pendingDot (row col k : Nat) (a b : Fin k → Expr)
  | queryDot (row col k : Nat) (a b : Fin k → Expr) (seed : Expr)
  | valueDot (row col k : Nat) (a b : Fin k → Expr) (previous alpha : Expr)
  /-- Generic backend ports absent from the admitted quotation remain marked.
  They cannot be mistaken for one of the claimed native instruction nodes. -/
  | unsupported

instance : Inhabited Expr := ⟨.bits 0⟩

def centerLike : Expr → Bool
  | .input rg _ => decide (rg="M")
  | .initialCenter | .center .. => Bool.true
  | .add a (.logarithm _) => centerLike a
  | _ => Bool.false

def consume : Binary → Expr → Expr → Expr
  | .add,.literal _,.negInf => .initialCenter
  | .add,.literal a,.literal b => .literal (a+b)
  | .add,.product previous alpha,.pendingDot row col k a b =>
      .valueDot row col k a b previous alpha
  | .add,seed,.pendingDot row col k a b => .queryDot row col k a b seed
  | .add,.product a b,acc => .fma a b acc
  | .add,a,b => .add a b
  | .mul,a,b => .product a b
  | .sub,a,b => if centerLike a then .centerDifference a b else .difference a b
  | .div,a,b => .divide a b
  | .max,a,b => .center a b
  | .pow,_,_ => .unsupported

def unary : Unary → Expr → Expr
  | .exp2,.centerDifference a b => .alpha a b
  | .exp2,a => .exponential a
  | .log2,a => .logarithm a
  | _,_ => .unsupported

def literal (x : ℝ) : Expr := if x=ScaleBinding.marker then .scale else .literal x

def matrix (batch : TileShape) {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr) :
    TileIndex (batch++[m,n]) → Expr :=
  match batch with
  | [] => fun i => .pendingDot i.1.val i.2.1.val k
      (fun j => a (i.1,j,())) (fun j => b (j,i.2.1,()))
  | _ :: tail => fun i => matrix tail (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2

def reduction (maximum : Bool) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (a : TileIndex shape → Expr)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  .reduction maximum (TileShape.linearIndex _ i) (TileShape.axisDim shape ax)
    (match keep with
      | Bool.false => fun j => a (TileShape.insertAxisIndex shape ax i j)
      | Bool.true => fun j => a (TileShape.replaceAxisIndex shape ax i j))

def algebra : Algebra Expr where
  literal := fun _ _ => literal
  negInf := .negInf
  binary := fun _ _ => consume
  unary := fun _ => unary
  cast := fun _ _ _ => id
  fromNat := fun _ _ => .unsupported
  fromInt := fun _ _ => .unsupported
  fp32Bits := fun x => .bits x.bits
  fp32Load := id
  reduceMax := fun _ {_} => reduction Bool.true
  reduceSum := fun _ {_} => reduction Bool.false

/-- The sole source floating comparison is tmp!=tmp. True equality here is
a staging token, so the source select transports its delayed alpha node.
Native.evaluate performs the NaN classification separately at every owner. -/
def backend : Reference.Ops Expr where
  scalar := algebra
  eq := fun _ _ _ _ => Bool.true
  dot := fun _ {batch} {_} {_} {_} => matrix batch
  dotOptions := fun _ _ {batch} {_} {_} {_} => matrix batch
  typedDot := fun _ l r out _ {batch} {_} {_} {_} a b i =>
    ofFloat out (matrix batch (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i)
  typedSum := fun _ d {_} ax keep a i => ofFloat d
    (reduction Bool.false ax keep (fun j => toFloat d (a j)) i)
  typedMax := fun _ d {_} ax keep a i => ofFloat d
    (reduction Bool.true ax keep (fun j => toFloat d (a j)) i)

theorem scale_leaf : literal ScaleBinding.marker=.scale := by simp [literal]
theorem alpha_staged (previous center : Expr) (tagged : centerLike previous=Bool.true) :
    (if !(backend.eq (some .fp32) .real
        (consume .sub previous center) (consume .sub previous center)) then literal 1
      else unary .exp2 (consume .sub previous center))=.alpha previous center := by
  simp [backend,consume,tagged,unary]

theorem source_fma (a b acc : Expr) : consume .add (consume .mul a b) acc=
    (match acc with
      | .pendingDot row col k left right => .valueDot row col k left right a b
      | _ => .fma a b acc) := by cases acc <;> rfl

theorem raw_center_payload_ignored (a b : BitVec 32) :
    centerLike (.input "M" a)=centerLike (.input "M" b) := rfl

#axiomsClean alpha_staged
#axiomsClean source_fma
#axiomsClean raw_center_payload_ignored
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
