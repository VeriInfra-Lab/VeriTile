import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.CSEBody

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
variable {α : Type} [Inhabited α]

theorem range (c : Call) (o : Ops α) (positive : 0<c.BN)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s) (memory : s.mem=initialState.mem)
    (cur stop : Int) :
    ∃ t,rangeInt (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) "start_n" cur stop c.BN (parts c).body s=some t ∧
      Matches c o (inputs initialState) (iterate c o (inputs initialState) cur stop old) t ∧
      t.mem=initialState.mem := by
  by_cases hc : cur<stop
  · obtain ⟨mid,he,live',memory'⟩ := body c o positive initialState s old live memory cur
    obtain ⟨t,ht,live'',memory''⟩ := range c o positive initialState mid
      (advance c o (inputs initialState) cur old) live' memory' (cur+c.BN) stop
    refine ⟨t,?_,?_,memory''⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt positive),if_pos hc,he]
      exact ht
    · rw [iterate,dif_neg (Nat.ne_of_gt positive),dif_pos hc]
      exact live''
  · refine ⟨s,?_,?_,memory⟩
    · rw [rangeInt,if_neg (Nat.ne_of_gt positive),if_neg hc]
    · rw [iterate,dif_neg (Nat.ne_of_gt positive),dif_neg hc]
      exact live
termination_by (stop-cur).toNat
decreasing_by omega

theorem loop (c : Call) (o : Ops α) (positive : 0<c.BN)
    (initialState s : State α) (live : Matches c o (inputs initialState) (initial c o (inputs initialState)) s)
    (memory : s.mem=initialState.mem) :
    ∃ t,step (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra)
      (.forRangeInt "start_n" (parts c).start (parts c).stop (parts c).stride (parts c).body) s=some t ∧
      Matches c o (inputs initialState) (result c o (inputs initialState)) t ∧
      t.mem=initialState.mem := by
  obtain ⟨t,ht,live',memory'⟩ := range c o positive initialState s (initial c o (inputs initialState))
    live memory (lower c (inputs initialState)) (upper c (inputs initialState))
  refine ⟨t,?_,live',memory'⟩
  rcases live with ⟨_,_,_,_,_,_,_,_,_,_,_,hlo,hhi⟩
  have lo : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).start s=
      some (fun _ => lower c (inputs initialState)) := hlo
  have hi : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).stop s=
      some (fun _ => upper c (inputs initialState)) := hhi
  have stride : evalOp (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) none (parts c).stride s=some (fun _ => c.BN) := rfl
  simpa only [step,lo,hi,stride,Option.bind_some] using ht

#axiomsClean range
#axiomsClean loop
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
