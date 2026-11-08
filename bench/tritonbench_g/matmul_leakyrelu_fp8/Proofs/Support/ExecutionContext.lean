import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Support.Execution

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false


theorem wide_count (x b : Nat) :
    (if (x : Int)<0 then -(-(x : Int)/(b : Int)) else (x : Int)/(b : Int)) +
      (if (x : Int)-(b : Int)*(if (x : Int)<0 then -(-(x : Int)/(b : Int))
        else (x : Int)/(b : Int))=0 then 0 else 1) =
      ((x/b + if x%b≠0 then 1 else 0 : Nat) : Int) := by
  simp only [not_lt.mpr (Int.natCast_nonneg x),↓reduceIte]
  rw [←Int.emod_def,←Int.natCast_mod]
  by_cases h:x%b=0 <;> simp [h,Int.natCast_dvd_natCast,Nat.dvd_iff_mod_eq_zero]

theorem wide_min (x y : Nat) :
    (if (x : Int)<(y : Int) then (x : Int) else (y : Int))=(min x y : Nat) := by
  by_cases h:x<y
  · simp [h,show (x : Int)<(y : Int) by exact_mod_cast h,Nat.min_eq_left (Nat.le_of_lt h)]
  · simp [h,show ¬(x : Int)<(y : Int) by exact_mod_cast h,Nat.min_eq_right (Nat.le_of_not_gt h)]

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Problem) (b : Buffers) (s : State α) :
    Context p s (preludeExecution M D p b s).val ∧
      HasAcc (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p b s).val := by
  have hm := wide_count p.M 128
  have hn := wide_count p.N 256
  have hk := wide_count p.K 64
  simp only [Nat.cast_ofNat] at hm hn hk
  change _ = (rows p : Int) at hm
  change _ = (columns p : Int) at hn
  change _ = (trips p : Int) at hk
  have hg : (if (8 : Int)<(rows p : Int) then 8 else (rows p : Int))=(group p : Int) := by
    simpa only [Nat.cast_ofNat,group] using wide_min 8 (rows p)
  constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,HasAcc]
  all_goals try rfl
  all_goals funext i
  all_goals simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,IntegerExpr.same_left,IntegerExpr.same_right]
  all_goals try simp only [Bool.not_eq_true_eq_eq_false,decide_eq_true_eq,decide_eq_false_iff_not,ite_not,add_zero,zero_add,zero_sub]
  all_goals try simp only [hm,hn,hk]
  all_goals try simp only [hg]
  all_goals try simp only [perGroup,firstM,groupSize,pidM,pidN,row,col,qDiv,qMod,min_form,Int.natCast_mul]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Execution
