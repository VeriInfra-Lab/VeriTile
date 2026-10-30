import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.ForwardLoop

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem epilogue (p : Ops α) (s : State α) (T D BD : Nat) (S : Bool)
    (carry : TileIndex [BD] → α)
    (hm : s.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j < D)))
    (hb : s.regs .nat [] "i_bh"=some (fun _ => s.pids 1))
    (hd : s.regs .nat [BD] "o_d"=some (lane s BD))
    (hc : s.regs .real [BD] "b_h"=some carry) :
    ∃ q, run (algebra p) (forwardEpilogue T D BD S) s=some q ∧
      q.mem=(if S then scatter s "ht" BD (slot s D BD)
        (fun j => lane s BD j < D) carry else s).mem ∧ q.pids=s.pids := by
  cases S <;>
    simp [forwardEpilogue,forward,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,FP.Structural.bop,
      natLt,numericLt,Region.cast,State.setReg,algebra,store,hm,hb,hd,hc,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat,scatter,slot,lane,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset]
  apply VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.scatter_memory_congr
  rfl

def ForwardWrites (s : State α) (T D BD : Nat) (S : Bool) (r : RegionName) (a : Nat) : Prop :=
  (r="o" ∧ ∃ t, t < T ∧ ∃ j, lane s BD j < D ∧ a=row s t T D BD j) ∨
  (S=Bool.true ∧ r="ht" ∧ ∃ j, lane s BD j < D ∧ a=slot s D BD j)

theorem forward_complete (p : Ops α) (s : State α) (T D BD : Nat) (U S : Bool) :
    ∃ q, run (algebra p) (forward T D BD U S).surfaceBody s=some q ∧ q.pids=s.pids ∧
      (∀ t, t < T → ∀ j, lane s BD j < D →
        q.mem "o" (row s t T D BD j)=.mk .real (values p s U T D BD (t+1) j)) ∧
      (S=Bool.true → ∀ j, lane s BD j < D →
        q.mem "ht" (slot s D BD j)=.mk .real (values p s U T D BD T j)) ∧
      (∀ r a, ¬ ForwardWrites s T D BD S r a → q.mem r a=s.mem r a) := by
  obtain ⟨q,he,P⟩ := forward_loop p s T D BD U
  have hl : lane q BD=lane s BD := by
    funext j
    simp [lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,P.pids]
  have hs : slot q D BD=slot s D BD := by
    funext j
    simp [slot,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,P.pids]
  obtain ⟨r,hr,hm,hp⟩ := epilogue p q T D BD S (values p s U T D BD T)
    (by simpa [hl] using P.mask) (by simpa [P.pids] using P.bh)
    (by simpa [hl] using P.channels) P.carry
  have memory : r.mem=(if S then scatter q "ht" BD (slot s D BD)
      (fun j => lane s BD j < D) (values p s U T D BD T) else q).mem := by
    simpa only [hl,hs] using hm
  refine ⟨r,?_,hp.trans P.pids,?_,?_,?_⟩
  · rw [forward_split,run_append,he]
    exact hr
  · intro t ht j hj
    rw [memory]
    cases S
    · exact P.history t ht j hj
    · rw [if_pos rfl,scatter_frame q "ht" BD _ _ _ "o" _ (Or.inl (by decide))]
      exact P.history t ht j hj
  · intro hS j hj
    subst S
    rw [memory,if_pos rfl]
    exact State.masked_scatter_readback "ht" (slot s D BD)
      (fun k => .mk .real (values p s U T D BD T k)) (fun k => lane s BD k < D) q j hj
      (by
        intro j k same
        apply VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset_injective s 0 1 D BD
        simpa [slot,row,
          VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset,
          VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset] using same)
  · intro rgn a miss
    have base : q.mem rgn a=s.mem rgn a := by
      apply P.frame rgn a
      by_cases h : rgn="o"
      · right
        intro t ht j hj ha
        exact miss (Or.inl ⟨h,t,ht,j,hj,ha⟩)
      · exact Or.inl h
    rw [memory]
    cases S
    · exact base
    · rw [if_pos rfl,scatter_frame]
      · exact base
      · by_cases h : rgn="ht"
        · right
          intro j hj ha
          exact miss (Or.inr ⟨rfl,h,j,hj,ha⟩)
        · exact Or.inl h

theorem values_eq_state (p : Ops α) (s : State α) (U : Bool) (T D BD : Nat)
    (input : Input α) (j : TileIndex [BD]) (hj : lane s BD j < D)
    (hx : ∀ t, read s "x" (row s t T D BD j)=input "x" (s.pids 1) t (lane s BD j))
    (hg : ∀ t, read s "g" (row s t T D BD j)=input "g" (s.pids 1) t (lane s BD j))
    (hh : read s "h0" (slot s D BD j)=input "h0" (s.pids 1) 0 (lane s BD j)) :
    ∀ t, values p s U T D BD t j=state p U input (s.pids 1) (lane s BD j) t := by
  intro t
  induction t with
  | zero => simp [values,initial,seed,state,masked,hj,hh]
  | succ t ih => simp [values,next,state,masked,hj,hx,hg,ih]

#axiomsClean forward_complete
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
