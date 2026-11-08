import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IndexProofs

@[simp] theorem product_address (a b : Nat) :
    ((a : Int)*(b : Int)).toNat=a*b := by
  simp only [← Int.natCast_mul,Int.toNat_natCast]

@[simp] theorem sum_address (a b : Nat) :
    ((a : Int)+(b : Int)).toNat=a+b := by
  simp only [← Int.natCast_add,Int.toNat_natCast]

@[simp] theorem product_sum_address (a b c : Nat) :
    ((a : Int)*(b : Int)+(c : Int)).toNat=a*b+c := by
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]

@[simp] theorem sum_lt (a b c : Nat) :
    (a : Int)+(b : Int)<(c : Int) ↔ a+b<c := by
  simp only [← Int.natCast_add,Int.ofNat_lt]

@[simp] theorem gathered_address (row cols pid block lane : Nat) :
    ((row : Int)*cols+((pid : Int)*block+lane)).toNat=row*cols+(pid*block+lane) := by
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast]

@[simp] theorem product_sum_lt (a b c d : Nat) :
    (a : Int)*b+c<(d : Int) ↔ a*b+c<d := by
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.ofNat_lt]

open VeriTile.Triton

/-- The actual signed loop visits a natural mathematical counter. The
invariant is transferred only after successful executions of its real body. -/
theorem range_safe {bounds : RegionBounds} {index : RegName} {stop stride : Nat}
    {body : List Stmt} {P : Nat → BlockState → Prop}
    (hsafe : ∀ i s, i<stop → P i s →
      Stmt.TraceSafeList bounds body (s.setReg index .int [] (Tile.scalar (i : Int))))
    (hnext : ∀ i s t, i<stop → P i s →
      stepStmts body (s.setReg index .int [] (Tile.scalar (i : Int)))=some t → P (i+stride) t)
    (cur : Nat) (s : BlockState) (hp : P cur s) :
    Stmt.forRangeIntTraceSafe bounds index cur stop stride body s := by
  rw [Stmt.forRangeIntTraceSafe]
  by_cases hz : stride=0
  · simp only [hz,if_true]
  · simp only [hz,if_false]
    by_cases hc : cur<stop
    · have hi : (cur : Int)<stop := by exact_mod_cast hc
      simp only [hi,if_true]
      refine ⟨hsafe cur s hc hp,?_⟩
      cases hb : stepStmts body (s.setReg index .int [] (Tile.scalar (cur : Int))) with
      | none => trivial
      | some t =>
        have next := range_safe hsafe hnext (cur+stride) t (hnext cur s t hc hp hb)
        simpa only [Nat.cast_add] using next
    · have hi : ¬(cur : Int)<stop := by exact_mod_cast hc
      simp only [hi,if_false]
termination_by stop-cur
 decreasing_by omega

#axiomsClean range_safe

end VeriTile.Bench.Repairs.FastRmsLayernorm.IndexProofs
