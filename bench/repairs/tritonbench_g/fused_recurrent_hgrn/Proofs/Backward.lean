import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.BackwardLoop
import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.Forward

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open VeriTile.Triton FP.Structural
namespace Execution
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

def BackwardWrites (s : State α) (T D BD : Nat) (r : RegionName) (a : Nat) : Prop :=
  (r="dx" ∨ r="dg") ∧ ∃ t, t < T ∧ ∃ j, lane s BD j < D ∧ a=row s t T D BD j

theorem backward_complete (p : Ops α) (s : State α) (T D BD : Nat) (U V : Bool) :
    ∃ q, run (algebra p) (backward T D BD U V).surfaceBody s=some q ∧ q.pids=s.pids ∧
      (∀ t, t < T → ∀ j, lane s BD j < D →
        q.mem "dx" (row s t T D BD j)=
          .mk .real (reverseDx p s T D BD t (reverseValues p s V T D BD (T-1-t)) j)) ∧
      (∀ t, t < T → ∀ j, lane s BD j < D →
        q.mem "dg" (row s t T D BD j)=
          .mk .real (reverseDg p s U T D BD t (reverseValues p s V T D BD (T-1-t)) j)) ∧
      (∀ r a, ¬ BackwardWrites s T D BD r a → q.mem r a=s.mem r a) := by
  obtain ⟨q,he,P⟩ := backward_loop p s T D BD U V
  refine ⟨q,he,P.pids,?_,?_,?_⟩
  · intro t ht j hj
    have heq : T-1-(T-1-t)=t := by omega
    simpa only [heq] using P.historyX (T-1-t) (by omega) j hj
  · intro t ht j hj
    have heq : T-1-(T-1-t)=t := by omega
    simpa only [heq] using P.historyG (T-1-t) (by omega) j hj
  · intro r a miss
    apply P.frame r a
    by_cases hr : r="dx" ∨ r="dg"
    · right
      intro m hm j hj ha
      exact miss ⟨hr,T-1-m,by omega,j,hj,ha⟩
    · left
      simpa only [not_or] using hr

theorem reverseValues_eq_adjoint (p : Ops α) (s : State α) (V : Bool) (T D BD : Nat)
    (input : Input α) (j : TileIndex [BD]) (hj : lane s BD j < D)
    (hdo : ∀ t, read s "do" (row s t T D BD j)=input "do" (s.pids 1) t (lane s BD j))
    (hg : ∀ t, read s "g" (row s t T D BD j)=input "g" (s.pids 1) t (lane s BD j))
    (hv : read s "dht" (slot s D BD j)=input "dht" (s.pids 1) 0 (lane s BD j)) :
    ∀ n, reverseValues p s V T D BD n j=adjoint p V input T (s.pids 1) (lane s BD j) n := by
  intro n
  induction n with
  | zero => simp [reverseValues,reverseInitial,adjoint,terminal,masked,hj,hv]
  | succ n ih => simp [reverseValues,reverseNext,reverseDx,adjoint,masked,hj,hdo,hg,ih]

theorem reverseDx_eq_dx (p : Ops α) (s : State α) (V : Bool) (T D BD t : Nat)
    (input : Input α) (j : TileIndex [BD]) (hj : lane s BD j < D)
    (hdo : ∀ t, read s "do" (row s t T D BD j)=input "do" (s.pids 1) t (lane s BD j))
    (hg : ∀ t, read s "g" (row s t T D BD j)=input "g" (s.pids 1) t (lane s BD j))
    (hv : read s "dht" (slot s D BD j)=input "dht" (s.pids 1) 0 (lane s BD j)) :
    reverseDx p s T D BD t (reverseValues p s V T D BD (T-1-t)) j=
      dx p V input T (s.pids 1) (lane s BD j) t := by
  simp [reverseDx,dx,masked,hj,hdo,reverseValues_eq_adjoint p s V T D BD input j hj hdo hg hv]

theorem reverseDg_eq_dg (p : Ops α) (s : State α) (U V : Bool) (T D BD t : Nat)
    (input : Input α) (j : TileIndex [BD]) (hj : lane s BD j < D) (ht : t < T)
    (hdo : ∀ t, read s "do" (row s t T D BD j)=input "do" (s.pids 1) t (lane s BD j))
    (hg : ∀ t, read s "g" (row s t T D BD j)=input "g" (s.pids 1) t (lane s BD j))
    (hv : read s "dht" (slot s D BD j)=input "dht" (s.pids 1) 0 (lane s BD j))
    (hh : read s "h0" (slot s D BD j)=input "h0" (s.pids 1) 0 (lane s BD j))
    (ho : ∀ k, k < T → read s "o" (row s k T D BD j)=state p U input (s.pids 1) (lane s BD j) (k+1)) :
    reverseDg p s U T D BD t (reverseValues p s V T D BD (T-1-t)) j=
      dg p U V input T (s.pids 1) (lane s BD j) t := by
  rw [reverseDg,dg,reverseDx_eq_dx p s V T D BD t input j hj hdo hg hv]
  congr 1
  by_cases hpos : 0 < t
  · have htm : t-1 < T := by omega
    have heq : t-1+1=t := by omega
    simp [reversePrevious,previous,masked,hj,hpos,ho _ htm,heq]
  · simp [reversePrevious,previous,masked,hj,hpos,hh]

#axiomsClean backward_complete
#specNonCircular dx avoiding [FP.Structural.exec]
#specNonCircular dg avoiding [FP.Structural.exec]
end
end Execution
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
