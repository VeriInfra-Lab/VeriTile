import bench.tritonbench_g.parallel_attention.Proofs.FPRange

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.RangeReuse
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 2000000

/-- A later typed register binding shadows the previous binding completely. -/
theorem setReg_shadow {α : Type} (s : State α) (name : RegName)
    (d e : TileDType) (sh th : TileShape) (v : Values α d sh) (w : Values α e th) :
    (s.setReg name d sh v).setReg name e th w = s.setReg name e th w := by
  cases s
  simp only [State.setReg]
  congr 1
  funext t dims reg
  by_cases h : reg = name <;> simp [h]

def signedBody (name : RegName) (body : List ComputeStmt) : List ComputeStmt :=
  .assign .int [] name (.alg (.castNatToInt (.ref .nat [] name))) :: body

theorem signedBody_run {α : Type} [Inhabited α] (M : Algebra α) (name : RegName)
    (body : List ComputeStmt) (s : State α) (n : Nat) :
    run M (signedBody name body) (s.setReg name .nat [] (fun _ => n)) =
      run M body (s.setReg name .int [] (fun _ => (n : Int))) := by
  simp [signedBody,run,step,evalExpr,evalOp_unfold,Option.bind,setReg_shadow]

/-- A signed nonnegative loop is related to the existing checked Nat loop by
one explicit Nat-to-Int register assignment. There is no body-result oracle. -/
theorem rangeInt_eq_range {α : Type} [Inhabited α] (M : Algebra α) (name : RegName)
    (body : List ComputeStmt) (s : State α) (cur stop stride : Nat) :
    rangeInt M name (cur : Int) (stop : Int) stride body s =
      FP.Structural.range M name cur stop stride (signedBody name body) s := by
  by_cases hz : stride = 0
  · simp [rangeInt,FP.Structural.range,hz]
  by_cases hc : cur < stop
  · have hi : (cur : Int) < (stop : Int) := by exact_mod_cast hc
    rw [rangeInt,if_neg hz,if_pos hi,FP.Structural.range,if_neg hz,if_pos hc,
      signedBody_run]
    cases he : run M body (s.setReg name .int [] (fun _ => (cur : Int))) with
    | none => simp [he]
    | some next =>
        simpa only [Nat.cast_add] using rangeInt_eq_range M name body next (cur+stride) stop stride
  · have hi : ¬(cur : Int) < (stop : Int) := by exact_mod_cast hc
    simp [rangeInt,FP.Structural.range,hz,hc,hi]
termination_by stop-cur
decreasing_by omega

/-- Reuse of the unchanged original full loop induction theorem. The new
source's signed K-block loop must still supply its actual one-body theorem. -/
theorem range_values_int {α β : Type} [Inhabited α] (M : Algebra α)
    (name : RegName) (body : List ComputeStmt) (update : Nat → β → β)
    (I : Nat → β → State α → Prop) (Done : β → State α → Prop)
    (cur stop stride : Nat) (initial : β) (s : State α)
    (hstep : ∀ n a t, I n a t → n < stop →
      ∃ u, run M body (t.setReg name .int [] (fun _ => (n : Int))) = some u ∧
        I (n+stride) (update n a) u)
    (hdone : ∀ n a t, I n a t → Done a t) (h : I cur initial s) :
    ∃ final, rangeInt M name (cur : Int) (stop : Int) stride body s = some final ∧
      Done (foldRange update cur stop stride initial) final := by
  rw [rangeInt_eq_range]
  apply range_values M name (signedBody name body) update I Done cur stop stride initial s
  · intro n a t ht hn
    simpa only [signedBody_run] using hstep n a t ht hn
  · exact hdone
  · exact h

#axiomsClean setReg_shadow
#axiomsClean signedBody_run
#axiomsClean rangeInt_eq_range
#axiomsClean range_values_int
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.RangeReuse
