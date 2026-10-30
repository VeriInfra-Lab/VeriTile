import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.BackwardExecution

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def reverseValues (p : Ops α) (s : State α) (V : Bool) (T D BD : Nat) : Nat → TileIndex [BD] → α
  | 0 => reverseInitial p s V D BD
  | n+1 => reverseNext p s T D BD (T-1-n) (reverseValues p s V T D BD n)

structure BwdInv (p : Ops α) (s : State α) (U V : Bool) (T D BD n : Nat) (q : State α) : Prop where
  pids : q.pids=s.pids
  mask : q.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j < D))
  bh : q.regs .nat [] "i_bh"=some (fun _ => s.pids 1)
  channels : q.regs .nat [BD] "o_d"=some (lane s BD)
  carry : q.regs .real [BD] "b_dh"=some (reverseValues p s V T D BD n)
  historyX : ∀ m, m < n → ∀ j, lane s BD j < D →
    q.mem "dx" (row s (T-1-m) T D BD j)=
      .mk .real (reverseDx p s T D BD (T-1-m) (reverseValues p s V T D BD m) j)
  historyG : ∀ m, m < n → ∀ j, lane s BD j < D →
    q.mem "dg" (row s (T-1-m) T D BD j)=
      .mk .real (reverseDg p s U T D BD (T-1-m) (reverseValues p s V T D BD m) j)
  frame : ∀ r a, ((r≠("dx" : RegionName) ∧ r≠("dg" : RegionName)) ∨
      ∀ m, m < n → ∀ j, lane s BD j < D → a≠row s (T-1-m) T D BD j) → q.mem r a=s.mem r a

theorem reverse_snapshot (p : Ops α) (s q : State α) (U : Bool) (T D BD t : Nat)
    (carry : TileIndex [BD] → α) (hp : q.pids=s.pids)
    (hdo : ∀ a, q.mem "do" a=s.mem "do" a) (hg : ∀ a, q.mem "g" a=s.mem "g" a)
    (ho : ∀ a, q.mem "o" a=s.mem "o" a) (hh : ∀ a, q.mem "h0" a=s.mem "h0" a) :
    reverseDx p q T D BD t carry=reverseDx p s T D BD t carry ∧
    reverseDg p q U T D BD t carry=reverseDg p s U T D BD t carry ∧
    reverseNext p q T D BD t carry=reverseNext p s T D BD t carry := by
  repeat' constructor
  all_goals funext j
  all_goals simp [reverseDx,reverseDg,reverseNext,reversePrevious,masked,read,row,lane,slot,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.stateOffset,hp,hdo,hg,ho,hh]

theorem backward_invariant_step (p : Ops α) (s q : State α) (U V : Bool) (T D BD n : Nat)
    (hn : n < T) (P : BwdInv p s U V T D BD n q) :
    ∃ r, run (algebra p) (backwardBody T D BD U) (q.setReg "step" .nat [] (fun _ => n))=some r ∧
      BwdInv p s U V T D BD (n+1) r := by
  let b := q.setReg "step" .nat [] (fun _ => n)
  have hp : b.pids=s.pids := P.pids
  have hl : lane b BD=lane s BD := by
    funext j
    simp [lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,hp]
  have hrw : ∀ t, row b t T D BD=row s t T D BD := by
    intro t
    funext j
    simp [row,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,hp]
  obtain ⟨r,he,hm,hpid,hmask,hbh,hod,hcarry⟩ :=
    backward_step p b T D BD n U (reverseValues p s V T D BD n)
      (by simp [b]) (by simpa [b,hl] using P.mask)
      (by simpa [b,P.pids] using P.bh) (by simpa [b,hl] using P.channels) (by simpa [b] using P.carry)
  have preserved : ∀ r a, r≠("dx" : RegionName) → r≠("dg" : RegionName) → b.mem r a=s.mem r a := by
    intro r a hx hg
    exact P.frame r a (Or.inl ⟨hx,hg⟩)
  obtain ⟨hvx,hvg,hvn⟩ := reverse_snapshot p s b U T D BD (T-1-n) (reverseValues p s V T D BD n) hp
    (fun a => preserved "do" a (by decide) (by decide)) (fun a => preserved "g" a (by decide) (by decide))
    (fun a => preserved "o" a (by decide) (by decide)) (fun a => preserved "h0" a (by decide) (by decide))
  let x := scatter b "dx" BD (row s (T-1-n) T D BD) (fun j => lane s BD j < D)
    (reverseDx p s T D BD (T-1-n) (reverseValues p s V T D BD n))
  have memory : r.mem=(scatter x "dg" BD (row s (T-1-n) T D BD) (fun j => lane s BD j < D)
      (reverseDg p s U T D BD (T-1-n) (reverseValues p s V T D BD n))).mem := by
    simpa only [hl,hrw,hvx,hvg] using hm
  refine ⟨r,he,⟨hpid.trans hp,?_,?_,?_,?_,?_,?_,?_⟩⟩
  · simpa [b,P.mask] using hmask
  · simpa [b,P.bh] using hbh
  · simpa [b,P.channels] using hod
  · simpa only [hvn,reverseValues] using hcarry
  · intro m hm j hj
    rw [memory,scatter_frame x "dg" BD _ _ _ "dx" _ (Or.inl (by decide))]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hm with hm|hm
    · rw [show x=scatter b "dx" BD (row s (T-1-n) T D BD) (fun j => lane s BD j < D)
        (reverseDx p s T D BD (T-1-n) (reverseValues p s V T D BD n)) from rfl,scatter_frame]
      · exact P.historyX m hm j hj
      · right
        intro k hk
        exact ne_of_gt (row_lt s T D BD (T-1-m) (T-1-n) k j (by omega) hk)
    · subst m
      exact State.masked_scatter_readback "dx" (row s (T-1-n) T D BD)
        (fun k => .mk .real (reverseDx p s T D BD (T-1-n) (reverseValues p s V T D BD n) k))
        (fun k => lane s BD k < D) b j hj
        (VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset_injective s (T-1-n) T D BD)
  · intro m hm j hj
    rw [memory]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hm with hm|hm
    · rw [scatter_frame]
      · rw [show x=scatter b "dx" BD (row s (T-1-n) T D BD) (fun j => lane s BD j < D)
          (reverseDx p s T D BD (T-1-n) (reverseValues p s V T D BD n)) from rfl,
          scatter_frame b "dx" BD _ _ _ "dg" _ (Or.inl (by decide))]
        exact P.historyG m hm j hj
      · right
        intro k hk
        exact ne_of_gt (row_lt s T D BD (T-1-m) (T-1-n) k j (by omega) hk)
    · subst m
      exact State.masked_scatter_readback "dg" (row s (T-1-n) T D BD)
        (fun k => .mk .real (reverseDg p s U T D BD (T-1-n) (reverseValues p s V T D BD n) k))
        (fun k => lane s BD k < D) x j hj
        (VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset_injective s (T-1-n) T D BD)
  · intro rgn a miss
    have mx : rgn≠("dx" : RegionName) ∨ ∀ j, lane s BD j < D → a≠row s (T-1-n) T D BD j := by
      rcases miss with h|h
      · exact Or.inl h.1
      · exact Or.inr (fun j hj => h n (by omega) j hj)
    have mg : rgn≠("dg" : RegionName) ∨ ∀ j, lane s BD j < D → a≠row s (T-1-n) T D BD j := by
      rcases miss with h|h
      · exact Or.inl h.2
      · exact Or.inr (fun j hj => h n (by omega) j hj)
    rw [memory,scatter_frame _ _ _ _ _ _ _ _ mg]
    change (scatter b "dx" BD _ _ _).mem rgn a=s.mem rgn a
    rw [scatter_frame _ _ _ _ _ _ _ _ mx]
    apply P.frame rgn a
    rcases miss with h|h
    · exact Or.inl h
    · exact Or.inr (fun m hm j hj => h m (by omega) j hj)

theorem backward_loop (p : Ops α) (s : State α) (T D BD : Nat) (U V : Bool) :
    ∃ q, run (algebra p) (backward T D BD U V).surfaceBody s=some q ∧
      BwdInv p s U V T D BD T q := by
  obtain ⟨q,he,hm,hp,hmask,hbh,hod,hcarry⟩ := backward_prologue p s T D BD V
  have hI : BwdInv p s U V T D BD 0 q := by
    refine ⟨hp,hmask,hbh,hod,hcarry,?_,?_,?_⟩
    · intro m hm
      omega
    · intro m hm
      omega
    · intro r a _
      rw [hm]
  obtain ⟨r,hr,hR⟩ := range_complete (algebra p) "step" (backwardBody T D BD U) T
    (BwdInv p s U V T D BD)
    (fun n hn q h => backward_invariant_step p s q U V T D BD n hn h) q 0 (by omega) hI
  refine ⟨r,?_,hR⟩
  rw [backward_split,run_append,he]
  simpa [run,step] using hr

#axiomsClean backward_loop
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
