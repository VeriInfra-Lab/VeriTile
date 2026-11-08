import bench.tritonbench_g.parallel_retention_attention.Proofs.SignedRange
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile Triton FP.Structural
noncomputable section

theorem rangeInt_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (body : List ComputeStmt)
    (I : Nat → State α → Prop) (cur stop stride : Nat)
    (bodySafe : ∀ n,n<stop → ∀ s,I n s →
      Safety.traceSafeList M bounds body (s.setReg idx .int [] (fun _=>(n : Int))))
    (advance : ∀ n,n<stop → ∀ s,I n s → ∀ t,
      run M body (s.setReg idx .int [] (fun _=>(n : Int)))=some t → I (n+stride) t)
    (s : State α) (h : I cur s) :
    Safety.rangeIntSafe M bounds idx (cur : Int) (stop : Int) stride body s := by
  by_cases zero : stride=0
  · simp [Safety.rangeIntSafe,zero]
  by_cases live : cur<stop
  · rw [Safety.rangeIntSafe]
    simp only [zero,show (cur : Int)<stop by exact_mod_cast live,if_false,if_true]
    refine ⟨bodySafe cur live s h,?_⟩
    cases ran : run M body (s.setReg idx .int [] (fun _=>(cur : Int))) with
    | none => trivial
    | some t =>
      have ready:=advance cur live s h t ran
      simpa only [←Int.natCast_add] using
        rangeInt_safe M bounds idx body I (cur+stride) stop stride bodySafe advance t ready
  · simp [Safety.rangeIntSafe,zero,show ¬(cur : Int)<stop by exact_mod_cast live]
termination_by stop-cur
decreasing_by omega

#axiomsClean rangeInt_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
