import bench.tritonbench_g.attention_kernel.Proofs.Output
import bench.tritonbench_g.attention_kernel.Proofs.RealMath
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Real
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

@[simp] theorem float_roundtrip {α : Type} (dtype : FloatDType) (a : α) :
    toFloat dtype (ofFloat dtype a)=a := by cases dtype <;> rfl

/-- Initial finite inputs, tagged with the actual host-selected storage dtype. -/
def Reads (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) : Prop :=
  ∀ r∈[p.Q,p.K,p.V,p.Bias],∀ off,
    s.mem r off=.mk p.storage.tile (ofFloat p.storage.float (some (μ r off)))

theorem read_word (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (h : Reads p s μ)
    (r : RegionName) (hr : r∈[p.Q,p.K,p.V,p.Bias]) (off : Nat) :
    toFloat p.storage.float ((s.mem r off).read p.storage.tile)=some (μ r off) := by
  rw [h r hr off]
  simp [TypedSource.Storage.tile]

def qAt (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (m d : Nat) : ℝ :=
  μ p.Q (base p s p.q+m*p.q.row+d*p.q.dim)
def kAt (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (n d : Nat) : ℝ :=
  μ p.K (base p s p.k+d*p.k.dim+n*p.k.row)
def vAt (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (n d : Nat) : ℝ :=
  μ p.V (base p s p.v+n*p.v.row+d*p.v.dim)
def bAt (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (m c : Nat) : ℝ :=
  μ p.Bias (base p s p.bias+m*p.bias.row+c)

theorem query_check (p : Parameters) (s : State R) (i : TileIndex [p.bm,p.dim]) :
    (qptr p s).inBounds (TileShape.indexToList _ i) [0]=decide (s.pids 0*p.bm+i.1.val<p.length) := by
  change (decide (s.pids 0*p.bm+i.1.val<p.length) && Bool.true)=_
  simp

theorem key_check (p : Parameters) (s : State R) (row : Nat) (i : TileIndex [p.dim,p.bn]) :
    (kptr p s row).inBounds (TileShape.indexToList _ i) [1]=decide (row+i.2.1.val<p.length) := by
  change (decide (row+i.2.1.val<p.length) && Bool.true)=_
  simp

theorem value_check (p : Parameters) (s : State R) (row : Nat) (i : TileIndex [p.bn,p.dim]) :
    (vptr p s row).inBounds (TileShape.indexToList _ i) [0]=decide (row+i.1.val<p.length) := by
  change (decide (row+i.1.val<p.length) && Bool.true)=_
  simp

theorem query_real (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (h : Reads p s μ)
    (i : TileIndex [p.bm,p.dim]) (hi : s.pids 0*p.bm+i.1.val<p.length) :
    query algebra p s i=some (qAt p s μ (s.pids 0*p.bm+i.1.val) i.2.1.val*(p.scale*1.44269504)) := by
  have hr:=read_word p s μ h p.Q (by simp) (base p s p.q+(s.pids 0*p.bm+i.1.val)*p.q.row+i.2.1.val*p.q.dim)
  simp [query,query_check,algebra,hi,hr,qAt]

theorem key_real (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [p.dim,p.bn]) (hj : i.2.1.val<p.length-row) :
    key algebra p s row i=some (kAt p s μ (row+i.2.1.val) i.1.val) := by
  have hr:=read_word p s μ h p.K (by simp) (base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row)
  have hj' : row+i.2.1.val<p.length := by omega
  simp [key,key_check,algebra,hj',hr,kAt]

theorem value_real (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [p.bn,p.dim]) :
    value algebra p s row i=some (if i.1.val<p.length-row then vAt p s μ (row+i.1.val) i.2.1.val else 0) := by
  have hr:=read_word p s μ h p.V (by simp) (base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim)
  by_cases hj : i.1.val<p.length-row
  · have hj' : row+i.1.val<p.length := by omega
    simp [value,value_check,algebra,hj,hj',hr,vAt]
  · have hj' : ¬row+i.1.val<p.length := by omega
    simp [value,value_check,algebra,hj,hj',hr,vAt]

theorem dot_real (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [p.bm,p.bn]) (hi : s.pids 0*p.bm+i.1.val<p.length)
    (hj : i.2.1.val<p.length-row) :
    dotModel.apply (some .fp32) p.storage.float p.storage.float .fp32 {} (batch:=[])
      (fun j=>ofFloat p.storage.float (query algebra p s j))
      (fun j=>ofFloat p.storage.float (key algebra p s row j)) i=
      some ((p.scale*1.44269504)*(Finset.range p.dim).sum
        (fun d=>qAt p s μ (s.pids 0*p.bm+i.1.val) d*kAt p s μ (row+i.2.1.val) d)) := by
  change sum p.dim (fun d=>WithBot.realMul
    (toFloat p.storage.float (ofFloat p.storage.float (query algebra p s (i.1,d,()))))
    (toFloat p.storage.float (ofFloat p.storage.float (key algebra p s row (d,i.2.1,())))))=_
  simp only [float_roundtrip]
  have hr : (fun d : Fin p.dim=>WithBot.realMul (query algebra p s (i.1,d,()))
      (key algebra p s row (d,i.2.1,())))=
      (fun d=>some ((qAt p s μ (s.pids 0*p.bm+i.1.val) d.val*(p.scale*1.44269504))*
        kAt p s μ (row+i.2.1.val) d.val)) := by
    funext d
    rw [query_real p s μ h (i.1,d,()) hi,key_real p s μ h row (d,i.2.1,()) hj]
    rfl
  rw [hr,sum_some]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun d=>(qAt p s μ (s.pids 0*p.bm+i.1.val) d*(p.scale*1.44269504))*
      kAt p s μ (row+i.2.1.val) d) p.dim,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro d hd
  ring

theorem score_real (p : Parameters) (s : State R) (μ : RegionName→ Nat→ℝ) (h : Reads p s μ)
    (row : Nat) (i : TileIndex [p.bm,p.bn]) (hi : s.pids 0*p.bm+i.1.val<p.length)
    (hbn : 0<p.bn) (aligned : row%p.bn=0) :
    scores algebra dotModel p s row i=
      if i.2.1.val<p.length-row then some
        (RealMath.score p.dim p.bn p.scale (qAt p s μ) (kAt p s μ) (bAt p s μ)
          (s.pids 0*p.bm+i.1.val) (row+i.2.1.val)) else none := by
  by_cases hj : i.2.1.val<p.length-row
  · have hb0:=read_word p s μ h p.Bias (by simp)
      (base p s p.bias+(s.pids 0*p.bm+i.1.val)*p.bias.row+row/p.bn)
    have hb1:=read_word p s μ h p.Bias (by simp)
      (base p s p.bias+(s.pids 0*p.bm+i.1.val)*p.bias.row+(i.2.1.val%64+64))
    have hkmod : (row+i.2.1.val)%p.bn=i.2.1.val := by
      simp [Nat.add_mod,aligned,Nat.mod_eq_of_lt i.2.1.isLt]
    have hkdiv : (row+i.2.1.val)/p.bn=row/p.bn := by
      have he : row=row/p.bn*p.bn := by
        have he:=Nat.div_add_mod row p.bn
        rw [Nat.mul_comm p.bn] at he
        omega
      conv_lhs => rw [he]
      rw [Nat.add_comm,Nat.add_mul_div_right _ _ hbn,Nat.div_eq_of_lt i.2.1.isLt]
      omega
    simp only [scores,if_pos hj]
    rw [dot_real p s μ h row i hi hj]
    simp only [biasDynamic,biasStatic,if_pos hi,hb0,hb1]
    simp [algebra,zero,RealMath.score,bAt,hkdiv,hkmod,Nat.add_comm]
  · simp [scores,hj,algebra]

#axiomsClean dot_real
#axiomsClean score_real
#specNonCircular Reads avoiding [FP.Structural.exec,accumulated]
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Real
