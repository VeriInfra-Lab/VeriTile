import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.Host
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.RealMath
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Real
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
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

def dotModel : Source.Dot R where
  apply:=fun p l r out options {batch} {_} {_} {_} a b=>dotValues p l r out options batch a b

/-- Total logical extension of finite initial input storage. Values outside
storage are irrelevant because the separate source access bounds exclude them. -/
def Reads (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) : Prop :=
  ∀ r∈[p.Q,p.K,p.V,p.QS,p.KS],∀ off,s.mem r off=.mk .real (some (μ r off))

def qAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (m d : Nat) : ℝ :=
  μ p.Q (base p s p.q+m*p.q.row+d*p.q.dim)
def kAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (n d : Nat) : ℝ :=
  μ p.K (base p s p.k+d*p.k.dim+n*p.k.row)
def vAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (n d : Nat) : ℝ :=
  μ p.V (base p s p.v+n*p.v.row+d*p.v.dim)
def qsAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (block : Nat) : ℝ :=
  μ p.QS (s.pids 1*((p.length+127)/128)+block)
def ksAt (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (block : Nat) : ℝ :=
  μ p.KS (s.pids 1*((p.length+63)/64)+block)

theorem query_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (i : TileIndex [128,128]) (hi : s.pids 0*128+i.1.val<p.length) :
    query algebra p s i=some (if i.2.1.val<96 then qAt p s μ (s.pids 0*128+i.1.val) i.2.1.val else 0) := by
  have hr:=h p.Q (by simp) (base p s p.q+(s.pids 0*128+i.1.val)*p.q.row+i.2.1.val*p.q.dim)
  by_cases hd : i.2.1.val<96 <;> simp [query,algebra,hi,hd,zero,hr,qAt]

theorem key_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [128,64]) (hj : i.2.1.val<p.length-row) :
    key algebra p s row i=some (if i.1.val<96 then kAt p s μ (row+i.2.1.val) i.1.val else 0) := by
  have hr:=h p.K (by simp) (base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row)
  by_cases hd : i.1.val<96 <;> simp [key,algebra,hj,hd,zero,hr,kAt]

theorem value_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [64,128]) :
    value algebra p s row i=some
      (if i.1.val<p.length-row ∧ i.2.1.val<96 then vAt p s μ (row+i.1.val) i.2.1.val else 0) := by
  have hr:=h p.V (by simp) (base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim)
  by_cases ha : i.1.val<p.length-row ∧ i.2.1.val<96 <;> simp [value,algebra,ha,zero,hr,vAt]

theorem dot_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [128,64]) (hi : s.pids 0*128+i.1.val<p.length)
    (hj : i.2.1.val<p.length-row) :
    dotModel.apply (some .fp32) .real .real .real {} (batch:=[])
      (query (α:=R) algebra p s) (key (α:=R) algebra p s row) i=
      some ((Finset.range 96).sum (fun d=>qAt p s μ (s.pids 0*128+i.1.val) d*kAt p s μ (row+i.2.1.val) d)) := by
  change sum 128 (fun d=>WithBot.realMul (query algebra p s (i.1,d,())) (key algebra p s row (d,i.2.1,())))=_
  have hr : (fun d : Fin 128=>WithBot.realMul (query algebra p s (i.1,d,())) (key algebra p s row (d,i.2.1,())))=
      (fun d=>some (if d.val<96 then qAt p s μ (s.pids 0*128+i.1.val) d.val*kAt p s μ (row+i.2.1.val) d.val else 0)) := by
    funext d
    rw [query_real p s μ h (i.1,d,()) hi,key_real p s μ h row (d,i.2.1,()) hj]
    by_cases hd : d.val<96 <;> simp [WithBot.realMul,hd]
  rw [hr,sum_some,Fin.sum_univ_eq_sum_range
    (fun d : Nat=>if d<96 then qAt p s μ (s.pids 0*128+i.1.val) d*kAt p s μ (row+i.2.1.val) d else 0) 128,
    RealMath.pad_head]

theorem score_real (p : Parameters) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [128,64]) (hi : s.pids 0*128+i.1.val<p.length)
    (aligned : row%64=0) :
    scores algebra dotModel p s row i=
      if i.2.1.val<p.length-row then some
        (RealMath.score (qAt p s μ) (kAt p s μ) (qsAt p s μ) (ksAt p s μ)
          (s.pids 0*128+i.1.val) (row+i.2.1.val)) else none := by
  by_cases hj : i.2.1.val<p.length-row
  · have hqs:=h p.QS (by simp) (s.pids 1*((p.length+127)/128)+s.pids 0)
    have hks:=h p.KS (by simp) (s.pids 1*((p.length+63)/64)+row/64)
    have hqidx : (s.pids 0*128+i.1.val)/128=s.pids 0 := by have:=i.1.isLt;omega
    have hkidx : (row+i.2.1.val)/64=row/64 := by have:=i.2.1.isLt;omega
    simp only [scores,if_pos hj]
    rw [dot_real p s μ h row i hi hj]
    simp [algebra,queryScale,keyScale,hqs,hks,RealMath.score,qsAt,ksAt,hqidx,hkidx]
  · simp [scores,hj,algebra]

#axiomsClean dot_real
#axiomsClean score_real
#specNonCircular Reads avoiding [FP.Structural.exec,accumulated]
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Real
