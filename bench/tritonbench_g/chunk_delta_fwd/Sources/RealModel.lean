import bench.tritonbench_g.chunk_delta_fwd.Sources.Program
import bench.tritonbench_g.chunk_delta_fwd.Sources.Mathematical_RealMath
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ChunkDeltaFwd
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000

/-- Mathematical interpretation of the operations used by this source.
Unused unary/reduction channels play no role in any theorem below. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
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
def dot : DotPrimitive ℝ where
  apply := fun _ l r out _ {batch} {_} {_} {_} a b => dotValues l r out batch a b

/-- Logical input arrays, pinned to the four independent input descriptors. -/
def arrays (p : Parameters) (head : Nat) (m : Memory ℝ) : RealMath.Inputs where
  key := fun t e => (m "key" (head*(p.T*p.K)+e+t*p.K)).read .real
  delta := fun t e => (m "weight" (head/p.H*p.wBatch+head%p.H*p.wHead+t*p.wTime+e*p.wDim)).read .real
  value := fun t j => (m "value" (head*(p.T*p.V)+t*p.V+j)).read .real
  seed := fun e j => (m "initial" (head*p.K*p.V+e*p.V+j)).read .real

/-- Independent window recurrence: width and stride are separate. This
preserves all rounded source visits when BC does not divide BT. -/
def state (xs : RealMath.Inputs) (T K stride width : Nat) (useSeed : Bool) :
    Nat → Nat → Nat → ℝ
  | 0,e,j => if useSeed then xs.seed e j else 0
  | c+1,e,j => state xs T K stride width useSeed c e j +
      ∑ t : Fin width, if c*stride+t.val < T then xs.key (c*stride+t.val) e *
        RealMath.corrected xs K (state xs T K stride width useSeed c) (c*stride+t.val) j else 0

def target (p : Parameters) (head : Nat) (m : Memory ℝ) :=
  state (arrays p head m) p.T p.K p.BT (innerCount p*p.BC) p.useSeed
def Represents (p : Parameters) (pid : PID) (h : HValues p ℝ) (s : Nat → Nat → ℝ) : Prop :=
  ∀ i : HIndex p, hActive p pid i → h i=s i.1.val (column p pid i.2.1.val)

theorem padded_sum (K BK : Nat) (h : K ≤ BK) (f : Nat → ℝ) :
    (∑ e : Fin BK, if e.val < K then f e.val else 0) = ∑ e : Fin K, f e.val := by
  rw [Fin.sum_univ_eq_sum_range (fun e => if e < K then f e else 0) BK,
    Fin.sum_univ_eq_sum_range f K, ← Finset.sum_filter]
  have hs : (Finset.range BK).filter (fun e => e < K) = Finset.range K := by
    ext e
    simp only [Finset.mem_filter,Finset.mem_range]
    omega
  rw [hs]

theorem corrected_expanded (p : Parameters) (pid : PID) (m : Memory ℝ)
    (c b : Nat) (h : HValues p ℝ) (i : CIndex p) :
    corrected algebra dot p pid m c b h i =
      valueLoad algebra p pid m c b i - ∑ e : Fin p.BK,
        weightLoad algebra p pid m c b ⟨i.1,e,()⟩ * h ⟨e,i.2.1,()⟩ := rfl

theorem corrected_eq (p : Parameters) (pid : PID) (m : Memory ℝ)
    (c b : Nat) (h : HValues p ℝ) (s : Nat → Nat → ℝ) (hK : p.K ≤ p.BK)
    (hh : Represents p pid h s) (i : CIndex p) (hi : cActive p pid c b i) :
    corrected algebra dot p pid m c b h i =
      RealMath.corrected (arrays p pid.2 m) p.K s (time p c b i.1.val) (column p pid i.2.1.val) := by
  rw [corrected_expanded]
  unfold RealMath.corrected
  congr 1
  · simp [valueLoad,hi.1,hi.2,algebra,arrays]
  · calc
      _ = ∑ e : Fin p.BK, if e.val < p.K then
          (arrays p pid.2 m).delta (time p c b i.1.val) e.val * s e.val (column p pid i.2.1.val) else 0 := by
        apply Finset.sum_congr rfl
        intro e _
        by_cases he : e.val < p.K
        · have hv := hh ⟨e,i.2.1,()⟩ ⟨he,hi.2⟩
          simp [weightLoad,hi.1,he,algebra,weightBase,arrays,hv]
        · simp [weightLoad,he,algebra,zero]
      _ = _ := padded_sum p.K p.BK hK (fun e =>
        (arrays p pid.2 m).delta (time p c b i.1.val) e * s e (column p pid i.2.1.val))

#axiomsClean corrected_eq
#specNonCircular state avoiding [run,FP.Structural.exec,states]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Real
