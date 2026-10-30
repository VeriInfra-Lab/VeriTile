import bench.tritonbench_g.matmul_dequantize.Proofs.FirstGrid
import bench.tritonbench_g.matmul_dequantize.Proofs.SecondGrid
import bench.tritonbench_g.matmul_dequantize.Proofs.DequantGrid32
import bench.tritonbench_g.matmul_dequantize.Proofs.DequantGrid16

namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Real
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The mathematical evaluator is explicit; this is not an FP-to-Real law. -/
def scalar : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | .max => max x y | .pow => 0
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceSum := fun _ _ _ _ _ => 0
  reduceMax := fun _ _ _ _ _ => 0

def sum (f : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => sum f n+f n

theorem sum_congr (f g : Nat → ℝ) (n : Nat) (h : ∀ k, k<n → f k=g k) : sum f n=sum g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sum,ih (fun k hk => h k (by omega)),h n (by omega)]

theorem fin_fold (f : Nat → ℝ) (n : Nat) (z : ℝ) :
    (List.finRange n).foldl (fun t i => t+f i.val) z=z+sum f n := by
  induction n with
  | zero => simp [sum]
  | succ n ih => simp [List.finRange_succ_last,List.foldl_append,List.foldl_map,ih,sum,add_assoc]

theorem sum_add (f : Nat → ℝ) (a b : Nat) : sum f (a+b)=sum f a+sum (fun k => f (a+k)) b := by
  induction b with
  | zero => simp [sum]
  | succ b ih => simp [sum,ih,Nat.add_assoc,add_assoc]

theorem sum_padding (f : Nat → ℝ) (K n : Nat) :
    sum (fun k => if k<K then f k else 0) n=sum f (min K n) := by
  induction n with
  | zero => simp [sum]
  | succ n ih =>
    by_cases hn : n<K
    · have hmin : min K n=n := Nat.min_eq_right (by omega)
      have hnext : min K (n+1)=n+1 := Nat.min_eq_right (by omega)
      simp [sum,ih,hn,hmin,hnext]
    · have hmin : min K n=K := Nat.min_eq_left (by omega)
      have hnext : min K (n+1)=K := Nat.min_eq_left (by omega)
      simp [sum,ih,hn,hmin,hnext]

def chunks (f : Nat → ℝ) (B : Nat) : Nat → ℝ
  | 0 => 0
  | n+1 => chunks f B n+sum (fun k => f (n*B+k)) B

theorem chunks_eq (f : Nat → ℝ) (B n : Nat) : chunks f B n=sum f (n*B) := by
  induction n with
  | zero => simp [chunks,sum]
  | succ n ih => simp only [chunks,ih,Nat.succ_mul,sum_add]

def dotValues : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → ℝ) → (TileIndex (batch++[k,n]) → ℝ) →
    (TileIndex (batch++[m,n]) → ℝ)
  | [],_,k,_,a,b,i => (List.finRange k).foldl (fun t e => t+a ⟨i.1,e,()⟩*b ⟨e,i.2.1,()⟩) 0
  | _::ds,_,_,_,a,b,i => dotValues ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2

def firstDot : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Dot ℝ where
  apply := fun _ _ {batch} {_} {_} {_} a b => dotValues batch a b

def secondDot : FPTypedSecond.Dot ℝ where
  apply := fun _ _ {batch} {_} {_} {_} a b => dotValues batch a b

def nibble (word : Int) (slot : Nat) : Int :=
  (((BitVec.ofInt 32 word) >>> (slot%8*4)) &&& (15 : BitVec 32)).toInt

theorem nibble_eq (word : Int) (slot : Nat) (h : -(2^31)≤word ∧ word<2^31) :
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble word slot=nibble word slot := by
  have hw : (BitVec.ofInt 32 word).toInt=word := BitVec.toInt_ofInt_eq_self (by decide) h.1 h.2
  simpa only [hw,nibble] using
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble_packed_word (BitVec.ofInt 32 word) slot

/-- Logical packed coordinates and the independent matrix product, unrelated
to tiling, schedules, source execution or final memory. -/
structure Inputs where
  a : Nat → Nat → ℝ
  weight : Nat → Nat → BitVec 32
  zero : Nat → Nat → BitVec 32
  scale : Nat → Nat → ℝ

def dequant (v : Inputs) (group k n : Nat) : ℝ :=
  let q := (((v.weight (k/8) n) >>> (k%8*4)) &&& (15 : BitVec 32)).toInt
  let z := (((v.zero (k/group) (n/8)) >>> (n%8*4)) &&& (15 : BitVec 32)).toInt
  (q-z : Int)*v.scale (k/group) n

def matmul (v : Inputs) (K group m n : Nat) : ℝ :=
  sum (fun k => v.a m k*dequant v group k n) K

def signedNibble (word : Int) (slot : Nat) : Int :=
  ((BitVec.ofInt 32 (word >>> (slot%8*4))) &&& (15 : BitVec 32)).toInt

theorem signedNibble_bits (word : BitVec 32) (slot : Nat) :
    signedNibble word.toInt slot=(((word >>> (slot%8*4)) &&& (15 : BitVec 32)).toInt) :=
  VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.nibble_packed_word word slot

structure RawInputs where
  a : Nat → Nat → ℝ
  weight : Nat → Nat → Int
  zero : Nat → Nat → Int
  scale : Nat → Nat → ℝ

def rawDequant (v : RawInputs) (group k n : Nat) : ℝ :=
  ((signedNibble (v.weight (k/8) n) k-signedNibble (v.zero (k/group) (n/8)) n : Int):ℝ)*v.scale (k/group) n

def rawMatmul (v : RawInputs) (K group m n : Nat) : ℝ :=
  sum (fun k => v.a m k*rawDequant v group k n) K

def Inputs.raw (v : Inputs) : RawInputs :=
  ⟨v.a,fun k n => (v.weight k n).toInt,fun g n => (v.zero g n).toInt,v.scale⟩

theorem rawDequant_bits (v : Inputs) (group k n : Nat) : rawDequant v.raw group k n=dequant v group k n := by
  simp only [rawDequant,dequant,Inputs.raw,signedNibble_bits]

theorem rawMatmul_bits (v : Inputs) (K group m n : Nat) : rawMatmul v.raw K group m n=matmul v K group m n := by
  unfold rawMatmul matmul
  apply sum_congr
  intro k hk
  rw [rawDequant_bits]
  rfl

theorem dot_sum (B : Nat) (a b : Fin B → ℝ) (f : Nat → ℝ) (h : ∀ e, a e*b e=f e.val) :
    (List.finRange B).foldl (fun t e => t+a e*b e) 0=sum f B := by
  have he : (fun t e => t+a e*b e)=(fun t (e : Fin B) => t+f e.val) := by
    funext t e
    rw [h e]
  rw [he,fin_fold,zero_add]

#axiomsClean fin_fold
#axiomsClean sum_padding
#axiomsClean chunks_eq
#axiomsClean nibble_eq
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Real
