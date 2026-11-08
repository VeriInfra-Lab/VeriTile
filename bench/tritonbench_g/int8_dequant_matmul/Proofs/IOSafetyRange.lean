import bench.tritonbench_g.int8_dequant_matmul.Proofs.IOSafetyLoop

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem loop_safe_iff (M : Algebra α) (arithmetic : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (s : Shape) (c : Config) (hasBias : Bool)
    (bounds : RegionName → Nat) (state : State α) :
    IntegerDot.stepSafe M arithmetic context bounds (loopStatement s c hasBias) state ↔
      IntegerDot.rangeIntSafe M arithmetic context bounds "k" 0 (steps s c) 1
        (body s c hasBias) state := by
  rw [loop_shape,IntegerDot.stepSafe_unfold]
  simp only [loop_bound_value]
  simp [evalOp_unfold,Safety.safeOp_unfold,loopBound]

/-- The shared safety checker follows every actual integer-dot successor.
The invariant and accumulator witness are the existing source execution facts. -/
theorem range_safe (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (base : State α)
    (bounds : RegionName → Nat)
    (ha : ∀ t, t<steps s c → ∀ i : TileIndex [c.BM,c.BK],
      (s.K%(c.BK*c.S)=0 ∨ register base .int [c.BK] "rk" (i.2.1,())<((s.K:Int)-(t*(c.BK*c.S):Nat))) →
      (ap s c base t i).2<bounds (ap s c base t i).1)
    (hb : ∀ t, t<steps s c → ∀ i : TileIndex [c.BK,c.BN],
      (s.K%(c.BK*c.S)=0 ∨ register base .int [c.BK] "rk" (i.1,())<((s.K:Int)-(t*(c.BK*c.S):Nat))) →
      (bp s c base t i).2<bounds (bp s c base t i).1)
    (t : Nat) (current : State α) (old : TileIndex [c.BM,c.BN] → Int)
    (ctx : Context s c base t current) (acc : HasAcc c old current) :
    IntegerDot.rangeIntSafe M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
      bounds "k" (t:Int) (steps s c) 1 (body s c hasBias) current := by
  by_cases active : t<steps s c
  · rw [IntegerDot.rangeIntSafe]
    simp only [Int.ofNat_lt,show 0<1 ∧ t<steps s c from ⟨by decide,active⟩,↓reduceIte]
    constructor
    · exact body_safe M _ _ s c hasBias base current t ctx bounds (ha t active) (hb t active)
    · rw [SharedInteger.run_eq,(bodyExecution M q s c hasBias base current t old ctx acc).property]
      have next := bodyContext M q s c hasBias base current t old ctx acc
      simpa only [Nat.cast_add,Nat.cast_one] using range_safe M q s c hasBias base bounds ha hb (t+1)
        (bodyExecution M q s c hasBias base current t old ctx acc).val
        (update q s c base t old) next.1 next.2
  · rw [IntegerDot.rangeIntSafe]
    simp only [Int.ofNat_lt,active,and_false,↓reduceIte]
termination_by steps s c-t
decreasing_by omega

#axiomsClean loop_safe_iff
#axiomsClean range_safe
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
