import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.IOSafetyLoop

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The shared safety checker follows every actual integer-dot successor.
The invariant and accumulator witness are the existing source execution facts. -/
theorem range_safe (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (base : State α)
    (bounds : RegionName → Nat)
    (ha : ∀ t, t<steps s c → ∀ i : TileIndex [c.BM,c.BK],
      (s.K%(c.BK*c.S)=0 ∨ register base .int [c.BK] "rk" (i.2.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (ap s c base t i).2<bounds (ap s c base t i).1)
    (hb : ∀ t, t<steps s c → ∀ i : TileIndex [c.BK,c.BN],
      (s.K%(c.BK*c.S)=0 ∨ register base .int [c.BK] "rk" (i.1,())<(s.K-t*(c.BK*c.S) : Nat)) →
      (bp s c base t i).2<bounds (bp s c base t i).1)
    (t : Nat) (current : State α) (old : TileIndex [c.BM,c.BN] → Int)
    (ctx : Context s c base t current) (acc : HasAcc c old current) :
    IntegerDot.rangeSafe M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
      bounds "k" t (steps s c) 1 (body s c hasBias) current := by
  by_cases active : t<steps s c
  · rw [IntegerDot.rangeSafe]
    simp only [show 0<1 ∧ t<steps s c from ⟨by decide,active⟩,↓reduceIte]
    constructor
    · exact body_safe M _ _ s c hasBias base current t ctx bounds (ha t active) (hb t active)
    · rw [SharedInteger.run_eq,(bodyExecution M q s c hasBias base current t old ctx acc).property]
      have next := bodyContext M q s c hasBias base current t old ctx acc
      exact range_safe M q s c hasBias base bounds ha hb (t+1)
        (bodyExecution M q s c hasBias base current t old ctx acc).val
        (update q s c base t old) next.1 next.2
  · rw [IntegerDot.rangeSafe]
    simp only [active,and_false,↓reduceIte]
termination_by steps s c-t
decreasing_by omega

#axiomsClean range_safe
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
