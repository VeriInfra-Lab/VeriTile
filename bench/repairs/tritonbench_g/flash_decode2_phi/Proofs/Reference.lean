import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Execution
import Mathlib.Analysis.SpecialFunctions.Exp

namespace VeriTile.Bench.Repairs.FlashDecode2.Reference
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

/-- `none` is genuine negative infinity. Only finite arithmetic and negative-
infinity subtraction/exponentiation reached from positive finite summaries are
used in the correctness theorem. No real number stands for negative infinity. -/
def finiteBinary (op : ℝ → ℝ → ℝ) : Option ℝ → Option ℝ → Option ℝ
  | some x,some y => some (op x y)
  | _,_ => none
def extendedMax : Option ℝ → Option ℝ → Option ℝ
  | none,y => y | x,none => x | some x,some y => some (max x y)
def ideal : Algebra (Option ℝ) where
  literal := fun _ _ x => some x
  negInf := none
  binary := fun _ _ op => match op with
    | .add => finiteBinary (·+·) | .sub => finiteBinary (·-·)
    | .mul => finiteBinary (·*·) | .div => finiteBinary (·/·)
    | .max => extendedMax | .pow => fun _ _ => none
  unary := fun _ op x => match op,x with
    | .exp,none => some 0 | .exp,some a => some (Real.exp a)
    | _,_ => none
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => none
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => none
  reduceSum := fun _ _ _ _ _ _ => none

/-- Independent unshifted weighted-summary formula. -/
def denominator (x : Nat → ℝ) (n : Nat) : ℝ := ∑ j ∈ Finset.range n, Real.exp (x j)
def numerator (x : Nat → ℝ) (v : Nat → ι → ℝ) (n : Nat) (k : ι) : ℝ :=
  ∑ j ∈ Finset.range n, Real.exp (x j)*v j k
def target (x : Nat → ℝ) (v : Nat → ι → ℝ) (n : Nat) (k : ι) : ℝ :=
  numerator x v n k / denominator x n

def merge (x : ℝ) (v : TileIndex [BD] → ℝ) (c : Carry (Option ℝ) BD) : Carry (Option ℝ) BD :=
  let m := maximum ideal (some x) c.m
  let old := exponential ideal (sub ideal c.m m)
  let fresh := exponential ideal (sub ideal (some x) m)
  ⟨m,add ideal (mul ideal c.z old) fresh,
    fun k => add ideal (mul ideal (c.a k) old) (mul ideal fresh (some (v k)))⟩
def recurrence (x : Nat → ℝ) (v : Nat → TileIndex [BD] → ℝ) : Nat → Carry (Option ℝ) BD
  | 0 => ⟨none,some 0,fun _ => some 0⟩
  | n+1 => merge (x n) (v n) (recurrence x v n)

theorem exp_rescale (a b : ℝ) : Real.exp (-a)*Real.exp (a-b)=Real.exp (-b) := by
  rw [←Real.exp_add]
  congr 1 <;> ring

theorem exp_weight (a b : ℝ) : Real.exp (a-b)=Real.exp a*Real.exp (-b) := by
  rw [←Real.exp_add]
  congr 1 <;> ring

theorem stabilized (x : Nat → ℝ) (v : Nat → TileIndex [BD] → ℝ)
    (n : Nat) (hn : 0<n) : ∃ m,
    (recurrence x v n).m=some m ∧
    (recurrence x v n).z=some (denominator x n*Real.exp (-m)) ∧
    ∀ k, (recurrence x v n).a k=some (numerator x v n k*Real.exp (-m)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases hz : n=0
    · subst n
      refine ⟨x 0,?_,?_,?_⟩
      · simp [recurrence,merge,maximum,ideal,extendedMax]
      · simp [recurrence,merge,maximum,exponential,sub,mul,add,ideal,extendedMax,finiteBinary,
          denominator,←Real.exp_add]
      · intro k
        simp [recurrence,merge,maximum,exponential,sub,mul,add,ideal,extendedMax,finiteBinary,
          numerator]
        rw [mul_right_comm,←Real.exp_add]
        simp
    · obtain ⟨m,hm,hz,ha⟩ := ih (by omega)
      let M := max (x n) m
      refine ⟨M,?_,?_,?_⟩
      · simp [recurrence,merge,maximum,ideal,extendedMax,hm,M]
      · simp only [recurrence,merge,hm,hz,maximum,exponential,sub,mul,add,ideal,
          extendedMax,finiteBinary]
        apply congrArg some
        change denominator x n * Real.exp (-m) * Real.exp (m-M) + Real.exp (x n-M) = _
        rw [mul_assoc,exp_rescale,exp_weight]
        simp [denominator,Finset.sum_range_succ,add_mul]
      · intro k
        simp only [recurrence,merge,hm,hz,ha,maximum,exponential,sub,mul,add,ideal,
          extendedMax,finiteBinary]
        apply congrArg some
        change numerator x v n k * Real.exp (-m) * Real.exp (m-M) + Real.exp (x n-M)*v n k = _
        rw [mul_assoc,exp_rescale,exp_weight]
        simp only [numerator,Finset.sum_range_succ]
        ring

theorem denominator_pos (x : Nat → ℝ) (n : Nat) (hn : 0<n) : 0<denominator x n := by
  apply Finset.sum_pos
  · intro i _
    exact Real.exp_pos _
  · exact ⟨0,Finset.mem_range.mpr hn⟩

theorem recurrence_correct (x : Nat → ℝ) (v : Nat → TileIndex [BD] → ℝ)
    (n : Nat) (hn : 0<n) (k : TileIndex [BD]) :
    divide ideal ((recurrence x v n).a k) (recurrence x v n).z=some (target x v n k) := by
  obtain ⟨m,hm,hz,ha⟩ := stabilized x v n hn
  simp only [divide,ideal,ha,hz,finiteBinary]
  apply congrArg some
  exact mul_div_mul_right _ _ (ne_of_gt (Real.exp_pos (-m)))

theorem source_recurrence (s : State (Option ℝ)) (masked : Bool) (H S D BD n : Nat)
    (x : Nat → ℝ) (v : Nat → TileIndex [BD] → ℝ)
    (hx : ∀ j, j<n → logvalue ideal s H S j=some (x j))
    (hv : ∀ j, j<n → ∀ k, value ideal s masked H S D j k=some (v j k)) :
    values ideal s masked H S D BD n=recurrence x v n := by
  induction n with
  | zero =>
    simp [values,initial,zero,ideal,recurrence]
  | succ n ih =>
    have h := ih (fun j hj => hx j (by omega)) (fun j hj => hv j (by omega))
    simp only [values,recurrence,h,next,merge,hx n (by omega)]
    congr 1
    funext k
    rw [hv n (by omega)]

theorem source_correct (s : State (Option ℝ)) (masked : Bool) (H S D Q BD : Nat)
    (x : Nat → ℝ) (v : Nat → TileIndex [BD] → ℝ)
    (positive : 0<count s Q)
    (hx : ∀ j, j<count s Q → logvalue ideal s H S j=some (x j))
    (hv : ∀ j, j<count s Q → ∀ k, value ideal s masked H S D j k=some (v j k))
    (k : TileIndex [BD]) : output ideal s masked H S D Q BD k=
      some (target x v (count s Q) k) := by
  unfold output
  rw [source_recurrence s masked H S D BD (count s Q) x v hx hv]
  exact recurrence_correct x v _ positive k

#axiomsClean source_correct
#axiomsClean denominator_pos
#specNonCircular target avoiding [FP.Structural.exec,values,recurrence]

end
end VeriTile.Bench.Repairs.FlashDecode2.Reference
