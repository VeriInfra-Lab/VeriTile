import bench.tritonbench_g.attention_fwd_triton3.Proofs.Program
import VeriTile.Triton.Math.Attention

/-! Exact mathematical interpretation of the actual source. Floating
precision and casts are erased to WithBot Real; integer casts and control
retain their source meaning. This model makes no IEEE numerical claim.
Bottom compares with itself in this mathematical projection, as in the
existing Real semantics; native NaN behavior remains in the separate FP IO. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
abbrev R := WithBot ℝ

/-- Exact, bottom-propagating finite sum. -/
def sum (n : Nat) (f : Fin n → R) : R :=
  if ∀ i,f i≠none then some (∑ i,(f i).unbotD 0) else none

@[simp] theorem sum_some (n : Nat) (f : Fin n → ℝ) :
    sum n (fun i=>some (f i))=some (∑ i,f i) := by simp [sum]

def exp2 : R → R
  | none=>some 0
  | some x=>some (pow2 x)

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values R .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (row : Fin (TileShape.axisDim shape ax)) : R :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i row)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i row)

def algebra : Algebra R where
  literal:=fun _ _ x=>some x
  negInf:=none
  binary:=fun _ _ op a b=>match op with
    | .add=>WithBot.realAdd a b
    | .sub=>WithBot.realSub a b
    | .mul=>WithBot.realMul a b
    | .div=>WithBot.realDiv a b
    | .max=>max a b
    | _=>none
  unary:=fun _ op a=>match op with
    | .exp2=>exp2 a
    | .log2=>a.map (fun x=>Real.log x/Real.log 2)
    | _=>none
  cast:=fun _ _ _ a=>a
  fromNat:=fun _ n=>some n
  fromInt:=fun _ n=>some n
  fp32Bits:=fun _=>none
  fp32Load:=id
  reduceMax:=fun _ {shape} ax keep xs i=>Finset.univ.sup (reductionInputs ax keep xs i)
  reduceSum:=fun _ {shape} ax keep xs i=>sum (TileShape.axisDim shape ax) (reductionInputs ax keep xs i)

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values R left.toTileDType (batch++[m,k]) →
      Values R right.toTileDType (batch++[k,n]) →
      Values R out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>ofFloat out (sum D (fun d=>
      WithBot.realMul (toFloat left (a (i.1,d,()))) (toFloat right (b (d,i.2.1,())))))
  | _::ds,_,_,_,a,b,i=>dotValues none left right out {} ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2


def ops : Reference.Ops R where
  scalar:=algebra
  eq:=fun _ _ a b=>decide (a=b)
  dot:=fun _ {batch} {_} {_} {_} a b=>dotValues none .real .real .real {} batch a b
  dotOptions:=fun _ cfg {batch} {_} {_} {_} a b=>dotValues none .real .real .real cfg batch a b
  typedDot:=fun _ l r out cfg {batch} {_} {_} {_} a b=>dotValues none l r out cfg batch a b
  typedSum:=fun _ dtype {shape} ax keep xs i=>ofFloat dtype
    (sum (TileShape.axisDim shape ax) (reductionInputs ax keep (fun j=>toFloat dtype (xs j)) i))
  typedMax:=fun _ dtype {shape} ax keep xs i=>ofFloat dtype
    (Finset.univ.sup (reductionInputs ax keep (fun j=>toFloat dtype (xs j)) i))

@[simp] theorem load_erasure (x : R) : ops.scalar.fp32Load x=x := rfl
@[simp] theorem cast_erasure (p : Option ComputeDType) (a b : FloatDType) (x : R) :
    ops.scalar.cast p a b x=x := rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
