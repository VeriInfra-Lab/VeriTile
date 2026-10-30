import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole001.Proofs.Reference
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole001.Proofs.Math
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Real
open VeriTile Triton FP.Structural Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
abbrev R := WithBot ℝ

def sum (n : Nat) (f : Fin n → R) : R :=
  if ∀ i,f i≠none then some (∑ i,(f i).unbotD 0) else none

@[simp] theorem sum_some (n : Nat) (f : Fin n → ℝ) :
    sum n (fun i=>some (f i))=some (∑ i,f i) := by simp [sum]

def exp2 : R → R | none=>some 0 | some x=>some (pow2 x)

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
  unary:=fun _ op a=>match op with | .exp2=>exp2 a | _=>none
  cast:=fun _ _ _ a=>a
  fromNat:=fun _ n=>some n
  fromInt:=fun _ n=>some n
  fp32Bits:=fun _=>none
  fp32Load:=id
  reduceMax:=fun _ {shape} ax keep xs i=>Finset.univ.sup (reductionInputs ax keep xs i)
  reduceSum:=fun _ {shape} ax keep xs i=>sum (TileShape.axisDim shape ax) (reductionInputs ax keep xs i)

def dotValues : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → R) → (TileIndex (batch++[k,n]) → R) →
    (TileIndex (batch++[m,n]) → R)
  | [],_,D,_,a,b,i=>sum D (fun d=>WithBot.realMul (a (i.1,d,())) (b (d,i.2.1,())))
  | _::ds,_,_,_,a,b,i=>dotValues ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dotModel : Dot R where
  equal:=fun a b=>decide (a=b)
  apply:=fun _ _ {batch} {_} {_} {_} a b=>dotValues batch a b

@[simp] theorem float_roundtrip {α : Type} (dtype : FloatDType) (a : α) :
    toFloat dtype (ofFloat dtype a)=a := by cases dtype <;> rfl

def Reads (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) : Prop :=
  ∀ r∈[c.q,c.k,c.v],∀ off,
    s.mem r off=.mk c.storage.tile (ofFloat c.storage.float (some (μ r off)))

theorem read_word (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (r : RegionName) (hr : r∈[c.q,c.k,c.v]) (off : Nat) :
    toFloat c.storage.float ((s.mem r off).read c.storage.tile)=some (μ r off) := by
  rw [h r hr off]
  simp [TypedSource.Storage.tile]

def qAt (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (row d : Nat) : ℝ :=
  μ c.q (queryBase c s+row*c.sqm+d*c.sqk)
def kAt (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (key d : Nat) : ℝ :=
  μ c.k (keyValueBase c s+d*c.skk+key*c.skn)
def vAt (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (key d : Nat) : ℝ :=
  μ c.v (keyValueBase c s+d*c.svk+key*c.svn)

theorem query_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (i : TileIndex [c.BM,c.BD]) (hi : rowStart c s+i.1.val<sequenceLength c s) :
    queryValues algebra c s i=some (qAt c s μ (rowStart c s+i.1.val) i.2.1.val*(c.scale*1.44269504)) := by
  simp only [queryValues,if_pos hi]
  rw [read_word c s μ h c.q (by simp)]
  rfl

theorem key_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : TileIndex [c.BD,c.BN]) :
    keyValues algebra c s ph start i=some
      (if keyActive c s ph start i.2.1 then kAt c s μ (columnValues c s ph start i.2.1) i.1.val else 0) := by
  by_cases hj : keyActive c s ph start i.2.1
  · simp only [keyValues,if_pos hj]
    rw [read_word c s μ h c.k (by simp)]
    rfl
  · norm_num [keyValues,hj,algebra]

theorem value_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (h : Reads c s μ)
    (ph : Bool) (start : Nat) (i : TileIndex [c.BN,c.BD]) :
    valueValues algebra c s ph start i=some
      (if keyActive c s ph start i.1 then vAt c s μ (columnValues c s ph start i.1) i.2.1.val else 0) := by
  by_cases hj : keyActive c s ph start i.1
  · simp only [valueValues,if_pos hj]
    rw [read_word c s μ h c.v (by simp)]
    rfl
  · norm_num [valueValues,hj,algebra]

#axiomsClean query_real
#axiomsClean key_real
#axiomsClean value_real
#specNonCircular Reads avoiding [FP.Structural.exec,finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.Real
