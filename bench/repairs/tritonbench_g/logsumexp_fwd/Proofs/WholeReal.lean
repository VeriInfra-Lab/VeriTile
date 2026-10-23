import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Real
import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Host

namespace VeriTile.Bench.Repairs.LogsumexpFwd.RealModel
open VeriTile.Triton TiledLogSumExp
noncomputable section
open Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def torchLSE (n : Nat) (values : Fin n → WithBot ℝ) : WithBot ℝ :=
  WithBot.realLog (∑ i,WithBot.realExp (values i))

def library : Host.Library (WithBot ℝ) (WithBot ℝ) := ⟨⊥,torchLSE,id⟩

/-- Source-independent host formula, including scalar/empty leading shapes.
The block cap is only an execution detail and disappears from this result. -/
theorem target_formula (a : Arguments) (input : Nat → ℝ)
    (row : Nat) (scale : ℝ)  :
    Host.target library a (fun _ => engine) (scale : WithBot ℝ)
      (fun i => (input i : WithBot ℝ)) row=
      formula input row (a.columns) a.hasScale scale := by
  by_cases hd : a.columns=0
  · simp [Host.target,formula,hd,library]
  · obtain ⟨n,hn⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt (block_positive a))
    have blocks : a.blocks=(a.columns+n)/(n+1) := by
      simp only [Arguments.blocks,hn]
      congr 1
    simp only [Host.target,if_neg hd]
    change torchLSE a.blocks _=_
    unfold torchLSE
    have each : ∀ b : Fin a.blocks,
        Grid.compute (fun _ => engine) a (scale : WithBot ℝ)
          (fun i => (input i : WithBot ℝ)) (row*a.blocks+b.val)=
          (partialLSE_full (fun i : Fin (a.columns) => input (row*a.columns+i.val)) b.val
            (by
              have bound : b.val<(a.columns+n)/(n+1) := by simpa only [blocks] using b.isLt
              exact h_tail_of_lt_cdiv ⟨b.val,bound⟩) a.hasScale scale : WithBot ℝ) := by
      intro b
      have positive : 0<a.blocks := Nat.lt_of_le_of_lt (Nat.zero_le _) b.isLt
      have rowEq : (row*a.blocks+b.val)/a.blocks=row := by
        rw [Nat.add_comm (row*a.blocks) b.val,Nat.add_mul_div_right _ _ positive,
          Nat.div_eq_of_lt b.isLt,Nat.zero_add]
      have blockEq : (row*a.blocks+b.val)%a.blocks=b.val := by
        simp [Nat.add_mod,Nat.mod_eq_of_lt b.isLt]
      unfold Grid.compute
      rw [rowEq,blockEq]
      change blockValue engine _ _ row b.val (a.columns) a.block a.hasScale=_
      rw [hn]
      exact block_real input row b.val (a.columns) n a.hasScale scale _
    simp_rw [each,WithBot.realExp_coe]
    rw [←WithBot.coe_sum,WithBot.realLog_coe]
    have changeIndex : (∑ b : Fin a.blocks,Real.exp
        (partialLSE_full (fun i : Fin (a.columns) => input (row*a.columns+i.val)) b.val
          (by
            have bound : b.val<(a.columns+n)/(n+1) := by simpa only [blocks] using b.isLt
            exact h_tail_of_lt_cdiv ⟨b.val,bound⟩) a.hasScale scale))=
      ∑ b : Fin ((a.columns+n)/(n+1)),Real.exp
        (partialLSE_full (fun i : Fin (a.columns) => input (row*a.columns+i.val)) b.val
          (h_tail_of_lt_cdiv b) a.hasScale scale) := by
      apply Fintype.sum_equiv (finCongr blocks)
      intro b
      rfl
    rw [changeIndex]
    exact aggregate_real input row (a.columns) n a.hasScale scale (Nat.pos_of_ne_zero hd)

#axiomsClean target_formula
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LogsumexpFwd.RealModel
