import bench.tritonbench_g.lightning_attention.Proofs.Target
import bench.tritonbench_g.lightning_attention.InterModel
import bench.tritonbench_g.lightning_attention.Proofs.HostMetadata
import bench.tritonbench_g.chunk_linear_attn.RealState
import bench.tritonbench_g.chunk_linear_attn.RealWindow
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  compareLe := fun _ _ => some (fun a b => decide (a ≤ b))
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b
    | .max => max a b | .pow => Real.rpow a b
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def dotValues (l r out : FloatDType) : (batch : TileShape) → {m k n : Nat} →
    Values ℝ l.toTileDType (batch++[m,k]) → Values ℝ r.toTileDType (batch++[k,n]) →
    Values ℝ out.toTileDType (batch++[m,n])
  | [],_,k,_,a,b,i => ofFloat out (∑ e : Fin k,
      toFloat l (a ⟨i.1,e,()⟩) * toFloat r (b ⟨e,i.2.1,()⟩))
  | _::ds,_,_,_,a,b,i => dotValues l r out ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) i.2
def dotPrimitive : DotPrimitive ℝ where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => dotValues l r out batch a b


def query (p : Parameters) (head : Nat) (m : Memory ℝ) (t a : Nat) : ℝ :=
  (m "query" (head*p.N*p.D+t*p.D+a)).read .real
def key (p : Parameters) (head : Nat) (m : Memory ℝ) (t a : Nat) : ℝ :=
  (m "key" (head*p.N*p.D+t*p.D+a)).read .real
def value (p : Parameters) (head : Nat) (m : Memory ℝ) (t c : Nat) : ℝ :=
  (m "value" (head*p.N*p.E+t*p.E+c)).read .real
def incoming (p : Parameters) (head : Nat) (m : Memory ℝ) (t c : Nat) : ℝ :=
  (m "incoming" (head*p.N*p.E+t*p.E+c)).read .real

def logicalQ (p : Parameters) (head : Nat) (m : Memory ℝ) : Fin p.N → Fin p.D → ℝ := fun t a => query p head m t a
def logicalK (p : Parameters) (head : Nat) (m : Memory ℝ) : Fin p.N → Fin p.D → ℝ := fun t a => key p head m t a
def logicalV (p : Parameters) (head : Nat) (m : Memory ℝ) : Fin p.N → Fin p.E → ℝ := fun t c => value p head m t c
def logicalG (p : Parameters) (head : Nat) (m : Memory ℝ) : Fin p.N → Fin p.E → ℝ := fun t c => incoming p head m t c

@[simp] theorem zero_eq : zero algebra=0 := rfl
@[simp] theorem add_eq (a b : ℝ) : add algebra a b=a+b := rfl
@[simp] theorem cast_eq (a : ℝ) : cast algebra a=a := rfl
@[simp] theorem read_eq (m : Memory ℝ) (r : RegionName) (offset : Nat)
    (active : Prop) [Decidable active] :
    read algebra m r offset active=(if active then (m r offset).read .real else 0) := rfl
@[simp] theorem dot_eq {M K N : Nat} (a : Matrix ℝ M K) (b : Matrix ℝ K N)
    (i : TileIndex [M,N]) : dot dotPrimitive a b i=
      ∑ k : Fin K, a (i.1,k,())*b (k,i.2.1,()) := rfl

theorem fold_sum {ι : Type} (f : Nat → ι → ℝ) (n : Nat) (z : ι → ℝ) (i : ι) :
    foldRange (fun sub acc j => acc j+f sub j) 0 n 1 z i=z i+∑ sub∈Finset.range n,f sub i := by
  induction n with
  | zero => simp [foldRange_zero]
  | succ n ih => rw [foldRange_succ]; rw [ih,Finset.sum_range_succ,add_assoc]

def blockSum (p : Parameters) (f : Nat → ℝ) (block : Nat) : ℝ :=
  ∑ lane : Fin p.BT, if block*p.BT+lane.val<p.N then f (block*p.BT+lane.val) else 0

theorem forward_update (p : Parameters) (pid : PID) (m : Memory ℝ) (n : Nat)
    (i : TileIndex [p.D,p.BM]) :
    dot dotPrimitive (fwdK algebra p pid m n) (fwdV algebra p pid m n) i=
      blockSum p (fun t => key p pid.head m t i.1.val*value p pid.head m t (pid.channel*p.BM+i.2.1.val)) n := by
  rw [dot_eq]
  unfold blockSum
  apply Finset.sum_congr rfl
  intro t _
  by_cases ht : n*p.BT+t.val<p.N <;>
    simp only [fwdK,fwdV,read_eq,ht,ite_true,ite_false,zero_mul,mul_zero,key,value]
  all_goals simp [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem forward_carry_blocks (p : Parameters) (pid : PID) (m : Memory ℝ) (n : Nat)
    (i : TileIndex [p.D,p.BM]) :
    fwdCarry algebra dotPrimitive p pid m n i=
      ∑ c∈Finset.range n,blockSum p (fun t => key p pid.head m t i.1.val*value p pid.head m t (pid.channel*p.BM+i.2.1.val)) c := by
  induction n with
  | zero => simp [fwdCarry]
  | succ n ih => simp only [fwdCarry,fwdNext,add_eq];rw [ih,forward_update,Finset.sum_range_succ]

theorem forward_carry (p : Parameters) (hp : 0<p.BT) (pid : PID) (m : Memory ℝ) (n : Nat)
    (i : TileIndex [p.D,p.BM]) :
    fwdCarry algebra dotPrimitive p pid m n i=
      ∑ t : Fin p.N, if t.val/p.BT<n then key p pid.head m t i.1.val*value p pid.head m t (pid.channel*p.BM+i.2.1.val) else 0 := by
  rw [forward_carry_blocks,← Fin.sum_univ_eq_sum_range]
  unfold blockSum
  rw [ChunkLinearAttn.Real.blocks_prefix p.N n p.BT (fun t => key p pid.head m t i.1.val*value p pid.head m t (pid.channel*p.BM+i.2.1.val))]
  apply Finset.sum_congr rfl
  intro t _
  simp only [Nat.div_lt_iff_lt_mul hp]

#axiomsClean forward_carry
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.Real
