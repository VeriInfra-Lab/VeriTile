import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedBodyValues
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem advance_context (M : Algebra α) (dot : Dot α) (c : Config) (initial : State α)
    (n start : Nat) (s : {s : State α // Ready c s})
    (ctx : Context M c initial (accumulated M dot c initial start) s.val) :
    Context M c initial (accumulated M dot c initial (start+n))
      (advance M dot c start n s).val := by
  induction n generalizing start s with
  | zero=>simpa [advance] using ctx
  | succ n ih=>
    have next := body_context M dot c initial s.val start (accumulated M dot c initial start) ctx s.property
    have hi := ih (start+1) ⟨body M dot c s.val s.property start,body_ready M dot c s.val s.property start⟩ next
    simpa only [advance,Nat.add_assoc,Nat.add_comm 1 n] using hi

theorem loop_context (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    Context M c s (accumulated M dot c s (valueBlocks c)) (loopState M dot c s).val := by
  simpa only [loopState,Nat.zero_add] using
    advance_context M dot c s (valueBlocks c) 0 ⟨(entry M dot c s).val,entry_ready M dot c s⟩
      (entry_context M dot c s)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
