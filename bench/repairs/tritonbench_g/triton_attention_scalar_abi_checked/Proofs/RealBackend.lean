import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardReadback
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.MathStreaming
import VeriTile.Triton.Math.Attention

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

abbrev R := WithBot ℝ
def sum (n : Nat) (f : Fin n → R) : R :=
  if ∀ i,f i≠none then some (∑ i,(f i).unbotD 0) else none
@[simp] theorem sum_some (n : Nat) (f : Fin n → ℝ) :
    sum n (fun i=>some (f i))=some (∑ i,f i) := by simp [sum]
def exp : R → R
  | none=>some 0
  | some x=>some (_root_.Real.exp x)
def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values R .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (j : Fin (TileShape.axisDim shape ax)) : R :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i j)
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
    | .exp=>exp a
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
def model : Primitives R where
  literal:=algebra.literal
  negInf:=algebra.negInf
  binary:=algebra.binary
  unary:=algebra.unary
  cast:=algebra.cast
  max:=algebra.reduceMax
  sum:=algebra.reduceSum
  dot:=fun precision l r out opts {batch} {_} {_} {_} a b=>dotValues precision l r out opts batch a b

#axiomsClean sum_some
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Erased
