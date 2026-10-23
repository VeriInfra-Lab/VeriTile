import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Reference
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.RealMath
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned Source VeriTile.Bench.Repairs.AttentionKernelAligned.Full VeriTile.Bench.Repairs.AttentionKernelAligned.FPValue
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
abbrev R := WithBot ℝ
def sum (n : Nat) (f : Fin n → R) : R :=
  if ∀ i,f i≠none then some (∑ i,(f i).unbotD 0) else none
@[simp] theorem sum_some (n : Nat) (f : Fin n → ℝ) :
    sum n (fun i=>some (f i))=some (∑ i,f i) := by simp [sum]
def exp2 : R → R
  | none=>some 0
  | some x=>some (pow2 x)
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
    | .exp2=>exp2 a
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
/-- Total extension of current finite FP16 input storage, without any result premise. -/
def Reads (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) : Prop :=
  ∀ r∈[p.q,p.k,p.v,p.bias],∀ off,s.mem r off=.mk .bf16 (some (μ r off))
def qAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (m d : Nat) : ℝ :=
  μ p.q (headBase p s+m*p.qRowStride+d*p.qColStride)
def kAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (n d : Nat) : ℝ :=
  μ p.k (kvBase p s+d*p.kDimStride+n*p.kSeqStride)
def vAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (n d : Nat) : ℝ :=
  μ p.v (valueBase p s+n*p.vSeqStride+d*p.vDimStride)
def biasAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (m j : Nat) : ℝ :=
  μ p.bias (biasBase p s+m*p.biasRowStride+j)
def logicalScore (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (i : Fin p.blockM) (j : Nat) : ℝ :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.RealMath.score p.head p.blockN p.biasLast p.scale (qAt p s μ) (kAt p s μ) (biasAt p s μ) (row p s i) j

theorem query_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : QLane p) (hi : row p s i.1<p.length) :
    qScaled model p s i=some (qAt p s μ (row p s i.1) i.2.1.val*(p.scale*1.44269504)) := by
  have hr:=h p.q (by simp) (qOffset p s i)
  dsimp only [qOffset] at hr
  simp [qScaled,qValue,scaleValue,model,algebra,hi,hr,qAt,qOffset,WithBot.realMul]
theorem key_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : KLane p) (hj : c*p.blockN+i.2.1.val<p.length) :
    kValue model p s c i=some (kAt p s μ (c*p.blockN+i.2.1.val) i.1.val) := by
  have hr:=h p.k (by simp) (kOffset p s c i)
  dsimp only [kOffset] at hr
  simp [kValue,hj,hr,kAt,kOffset]
theorem value_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : VLane p) :
    vValue model p s c i=some (if c*p.blockN+i.1.val<p.length then vAt p s μ (c*p.blockN+i.1.val) i.2.1.val else 0) := by
  have hr:=h p.v (by simp) (vOffset p s c i)
  dsimp only [vOffset] at hr
  by_cases hj : c*p.blockN+i.1.val<p.length <;> simp [vValue,hj,hr,vAt,vOffset,model,algebra]
theorem dot_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : ScoreLane p) (hi : row p s i.1<p.length)
    (hj : c*p.blockN+i.2.1.val<p.length) :
    qkDot model p s c i=some (∑ d∈Finset.range p.head,
      (qAt p s μ (row p s i.1) d*(p.scale*1.44269504))*kAt p s μ (c*p.blockN+i.2.1.val) d) := by
  change sum p.head (fun d=>WithBot.realMul (qScaled model p s (i.1,d,())) (kValue model p s c (d,i.2.1,())))=_
  have he : (fun d : Fin p.head=>WithBot.realMul (qScaled model p s (i.1,d,())) (kValue model p s c (d,i.2.1,())))=
      (fun d=>some ((qAt p s μ (row p s i.1) d.val*(p.scale*1.44269504))*kAt p s μ (c*p.blockN+i.2.1.val) d.val)) := by
    funext d
    rw [query_real p s μ h (i.1,d,()) hi,key_real p s μ h c (d,i.2.1,()) hj]
    rfl
  rw [he,sum_some]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun d : Nat => (qAt p s μ (row p s i.1) d*(p.scale*1.44269504))*kAt p s μ (c*p.blockN+i.2.1.val) d) p.head
theorem score_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (c : Nat) (i : ScoreLane p) (hi : row p s i.1<p.length) (hBN : 0<p.blockN) :
    score32 model p s c i=
      if c*p.blockN+i.2.1.val<p.length then some (logicalScore p s μ i.1 (c*p.blockN+i.2.1.val)) else none := by
  by_cases hj : c*p.blockN+i.2.1.val<p.length
  · have hb0:=h p.bias (by simp) (b0Offset p s c i.1)
    have hb1:=h p.bias (by simp) (b1Offset p s i)
    dsimp only [b0Offset] at hb0
    dsimp only [b1Offset] at hb1
    have hc : (c*p.blockN+i.2.1.val)/p.blockN=c := by
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ hBN,Nat.div_eq_of_lt i.2.1.isLt]
      omega
    have hm : (c*p.blockN+i.2.1.val)%p.blockN=i.2.1.val := by
      simp [Nat.add_mod,Nat.mod_eq_of_lt i.2.1.isLt]
    simp only [score32,score,if_pos hj,rawScore]
    rw [dot_real p s μ h c i hi hj]
    simp [model,algebra,halfZero,b0Value,b1Value,hi,hb0,hb1,logicalScore,VeriTile.Bench.Repairs.AttentionKernelAligned.RealMath.score,biasAt,b0Offset,b1Offset,hc,hm,WithBot.realAdd]
  · simp [score32,score,hj,model,algebra]
#axiomsClean dot_real
#axiomsClean score_real
#specNonCircular Reads avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Real
