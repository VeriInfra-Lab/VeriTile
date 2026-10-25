import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Execution
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Reference

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
variable {α : Type} [Inhabited α]

 theorem range_complete (M : Algebra α) (idx : RegName) (body : List ComputeStmt)
    (stop : Nat) (P : Nat → State α → Prop)
    (one : ∀ i, i < stop → ∀ s, P i s → ∃ q,
      run M body (s.setReg idx .nat [] (fun _ => i))=some q ∧ P (i+1) q)
    (s : State α) (start : Nat) (bound : start≤stop) (initial : P start s) :
    ∃ q, range M idx start stop 1 body s=some q ∧ P stop q := by
  have go : ∀ n i, i+n=stop → ∀ s, P i s →
      ∃ q, range M idx i stop 1 body s=some q ∧ P stop q := by
    intro n
    induction n with
    | zero =>
      intro i hi s hs
      have h : i=stop := by omega
      subst i
      exact ⟨s,by rw [range];simp,hs⟩
    | succ n ih =>
      intro i hi s hs
      have hlt : i < stop := by omega
      obtain ⟨q,hq,hP⟩ := one i hlt s hs
      obtain ⟨r,hr,hR⟩ := ih (i+1) (by omega) q hP
      refine ⟨r,?_,hR⟩
      rw [range]
      simpa [hlt,hq] using hr
  exact go (stop-start) start (by omega) s initial

 theorem scatter_frame (s : State α) (R : RegionName) (BD : Nat)
    (off : TileIndex [BD] → Nat) (active : TileIndex [BD] → Prop) [DecidablePred active]
    (value : TileIndex [BD] → α) (r : RegionName) (a : Nat)
    (miss : r≠R ∨ ∀ j, active j → a≠off j) :
    (scatter s R BD off active value).mem r a=s.mem r a := by
  apply State.masked_scatter_frame
  rcases miss with h|h
  · exact Or.inl h
  · exact Or.inr (fun j _ hj => h j hj)

 theorem row_lt (s : State α) (T D BD i t : Nat) (j k : TileIndex [BD])
    (ht : t < i) (hj : lane s BD j < D) : row s t T D BD j < row s i T D BD k := by
  dsimp [row,lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel] at *
  nlinarith

 def values (p : Ops α) (s : State α) (U : Bool) (T D BD : Nat) : Nat → TileIndex [BD] → α
  | 0 => initial p s U D BD
  | n+1 => next p s T D BD n (values p s U T D BD n)

 structure FwdInv (p : Ops α) (s : State α) (U : Bool) (T D BD i : Nat) (q : State α) : Prop where
  pids : q.pids=s.pids
  mask : q.regs .bool [BD] "mask"=some (fun j => decide (lane s BD j < D))
  bh : q.regs .nat [] "i_bh"=some (fun _ => s.pids 1)
  channels : q.regs .nat [BD] "o_d"=some (lane s BD)
  px : q.regs .ptr [BD] "p_x"=some (fun j => (("x" : RegionName),row s i T D BD j))
  pg : q.regs .ptr [BD] "p_g"=some (fun j => (("g" : RegionName),row s i T D BD j))
  po : q.regs .ptr [BD] "p_o"=some (fun j => (("o" : RegionName),row s i T D BD j))
  carry : q.regs .real [BD] "b_h"=some (values p s U T D BD i)
  history : ∀ t, t < i → ∀ j, lane s BD j < D →
    q.mem "o" (row s t T D BD j)=.mk .real (values p s U T D BD (t+1) j)
  frame : ∀ r a, (r≠("o" : RegionName) ∨ ∀ t, t < i → ∀ j, lane s BD j < D → a≠row s t T D BD j) →
    q.mem r a=s.mem r a

 theorem next_snapshot (p : Ops α) (s q : State α) (T D BD t : Nat)
    (carry : TileIndex [BD] → α) (hp : q.pids=s.pids)
    (hx : ∀ a, q.mem "x" a=s.mem "x" a) (hg : ∀ a, q.mem "g" a=s.mem "g" a) :
    next p q T D BD t carry=next p s T D BD t carry := by
  funext j
  simp [next,masked,read,row,lane,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
    VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,hp,hx,hg]

 theorem forward_invariant_step (p : Ops α) (s q : State α) (U : Bool) (T D BD i : Nat)
    (P : FwdInv p s U T D BD i q) :
    ∃ r, run (algebra p) (forwardBody D BD) (q.setReg "_" .nat [] (fun _ => i))=some r ∧
      FwdInv p s U T D BD (i+1) r := by
  let b := q.setReg "_" .nat [] (fun _ => i)
  have hp : b.pids=s.pids := P.pids
  have hl : lane b BD=lane s BD := by
    funext j
    simp [lane,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,hp]
  have hrw : ∀ t, row b t T D BD=row s t T D BD := by
    intro t
    funext j
    simp [row,VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset,
      VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.channel,hp]
  obtain ⟨r,he,hm,hpid,hmask,hbh,hod,hx,hg,ho,hcarry⟩ :=
    forward_step p b T D BD i (values p s U T D BD i)
      (by simpa [b,hl] using P.mask) (by simpa [b,hrw] using P.px)
      (by simpa [b,hrw] using P.pg) (by simpa [b,hrw] using P.po)
      (by simpa [b] using P.carry)
  have hv : next p b T D BD i (values p s U T D BD i)=values p s U T D BD (i+1) := by
    apply next_snapshot p s b T D BD i _ hp
    · intro a
      exact P.frame "x" a (Or.inl (by decide))
    · intro a
      exact P.frame "g" a (Or.inl (by decide))
  have memory : r.mem=(scatter b "o" BD (row s i T D BD)
      (fun j => lane s BD j < D) (values p s U T D BD (i+1))).mem := by
    simpa only [hl,hrw,hv] using hm
  refine ⟨r,he,⟨hpid.trans hp,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩⟩
  · simpa [b,P.mask] using hmask
  · simpa [b,P.bh] using hbh
  · simpa [b,P.channels] using hod
  · simpa only [hrw] using hx
  · simpa only [hrw] using hg
  · simpa only [hrw] using ho
  · simpa only [hv] using hcarry
  · intro t ht j hj
    rcases Nat.lt_succ_iff_lt_or_eq.mp ht with ht|ht
    · rw [memory,scatter_frame]
      · exact P.history t ht j hj
      · right
        intro k _
        exact ne_of_lt (row_lt s T D BD i t j k ht hj)
    · subst t
      rw [memory]
      exact State.masked_scatter_readback "o" (row s i T D BD)
        (fun k => .mk .real (values p s U T D BD (i+1) k))
        (fun k => lane s BD k < D) b j hj
        (VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn.FPStages.rowOffset_injective s i T D BD)
  · intro rgn a miss
    rw [memory,scatter_frame]
    · apply P.frame rgn a
      rcases miss with h|h
      · exact Or.inl h
      · exact Or.inr (fun t ht j hj => h t (by omega) j hj)
    · rcases miss with h|h
      · exact Or.inl h
      · exact Or.inr (fun j hj => h i (by omega) j hj)

 theorem forward_loop (p : Ops α) (s : State α) (T D BD : Nat) (U : Bool) :
    ∃ q, run (algebra p) (forwardPrologue T D BD U ++
      [.forRange "_" 0 T 1 (forwardBody D BD)]) s=some q ∧ FwdInv p s U T D BD T q := by
  obtain ⟨q,he,hm,hp,hmask,hbh,hod,hx,hg,ho,hcarry⟩ := prologue p s T D BD U
  have hI : FwdInv p s U T D BD 0 q := by
    refine ⟨hp,hmask,hbh,hod,hx,hg,ho,hcarry,?_,?_⟩
    · intro t ht
      omega
    · intro r a _
      rw [hm]
  obtain ⟨r,hr,hR⟩ := range_complete (algebra p) "_" (forwardBody D BD) T
    (FwdInv p s U T D BD) (fun i _ q h => forward_invariant_step p s q U T D BD i h)
    q 0 (by omega) hI
  refine ⟨r,?_,hR⟩
  rw [run_append,he]
  simpa [run,step] using hr

#axiomsClean forward_loop
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
