import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedAccum
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

def advance (M : Algebra α) (dot : Dot α) (c : Config) (start : Nat) :
    Nat → {s : State α // Ready c s} → {s : State α // Ready c s}
  | 0,s=>s
  | n+1,s=>advance M dot c (start+1) n
      ⟨body M dot c s.val s.property start,body_ready M dot c s.val s.property start⟩

theorem advance_runs (M : Algebra α) (dot : Dot α) (c : Config) (n start : Nat)
    (s : {s : State α // Ready c s}) :
    range (engine (withDot M dot)) "i_v" start (start+n) 1 (parts c).body s.val=
      some (advance M dot c start n s).val := by
  induction n generalizing start s with
  | zero=>rw [range];simp [advance]
  | succ n ih=>
    rw [range]
    simp only [show (1:Nat)≠0 from by decide,if_false,
      show start<start+(n+1) from by omega,if_true,
      body_runs M dot c s.val s.property start,bind,Option.bind_some]
    simpa only [advance,Nat.add_assoc,Nat.add_comm 1 n] using
      ih (start+1) ⟨body M dot c s.val s.property start,body_ready M dot c s.val s.property start⟩

theorem advance_memory (M : Algebra α) (dot : Dot α) (c : Config) (n start : Nat)
    (s : {s : State α // Ready c s}) : (advance M dot c start n s).val.mem=s.val.mem := by
  induction n generalizing start s with
  | zero=>rfl
  | succ n ih=>
    rw [advance,ih]
    exact body_memory M dot c s.val s.property start

def valueBlocks (c : Config) := (c.V+c.BV-1)/c.BV

def loopState (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    {t : State α // Ready c t} :=
  advance M dot c 0 (valueBlocks c) ⟨(entry M dot c s).val,entry_ready M dot c s⟩

theorem loop_runs (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    step (engine (withDot M dot))
      (.forRangeDyn "i_v" (parts c).start (parts c).stop (parts c).stride (parts c).body)
      (entry M dot c s).val=some (loopState M dot c s).val := by
  rw [step.eq_7]
  change range (engine (withDot M dot)) "i_v" 0 (valueBlocks c) 1 (parts c).body (entry M dot c s).val=_
  simpa only [Nat.zero_add,loopState] using advance_runs M dot c (valueBlocks c) 0 ⟨(entry M dot c s).val,entry_ready M dot c s⟩

theorem loop_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    (loopState M dot c s).val.mem=s.mem := by
  unfold loopState
  rw [advance_memory,entry_memory]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
