import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Runtime

namespace VeriTile.Bench.Repairs.KldivTriton.Launch
open Host
open L2NormBwd.Physical (Matrix)
open L2NormBwd.Integer (Metadata GridCompatible pointer)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
set_option autoImplicit false
set_option maxHeartbeats 3000000

theorem block_profile (N : Nat) : ∃ k,k≤14 ∧ blockSize N=2^k := by
  have positive : 0<2^Nat.clog 2 N := pow_pos (by decide) _
  by_cases small : Nat.clog 2 N≤14
  · have bound : 2^Nat.clog 2 N≤16384 := by
      have h := (Nat.pow_le_pow_iff_right (by decide : 1<2)).mpr small
      norm_num at h
      exact h
    exact ⟨Nat.clog 2 N,small,by simp [blockSize,Nat.min_eq_right bound,Nat.max_eq_right (by omega : 1≤2^Nat.clog 2 N)]⟩
  · have bound : 16384≤2^Nat.clog 2 N := by
      have h := (Nat.pow_le_pow_iff_right (by decide : 1<2)).mpr (show 14≤Nat.clog 2 N by omega)
      norm_num at h
      exact h
    exact ⟨14,by omega,by simp [blockSize,Nat.min_eq_left bound]⟩

theorem block_bound (N : Nat) : blockSize N≤16384 := by unfold blockSize;omega

/-- Ordinary launcher/metadata boundary, separate from numerical obligations.
A rejected setup is an ordinary API error. This does not impose a new accepted
logical shape cap or assume any successful kernel/output calculation. -/
structure Contract {M : Type} (rt : Runtime M) : Prop where
  forward : ∀ p queue before after,rt.perform (.setupForward p queue) before=.ok () after →
    Metadata p.prediction ∧ Metadata p.target ∧ Metadata p.output ∧ GridCompatible p.prediction.rows
  backward : ∀ p queue before after,rt.perform (.setupBackward p queue) before=.ok () after →
    Metadata p.target ∧ Metadata p.work ∧ GridCompatible p.work.rows

/-- All current/tail offsets and the terminal increment are exact signed64;
each active pointer covers four backed bytes. Bounds come from actual storage. -/
theorem prepared_indices (m : Matrix) (metadata : Metadata m) (backed : m.backed)
    (unit : Index.UnitColumns m) (grid : GridCompatible m.rows) (r c k lane : Nat)
    (hr : r<m.rows) (hc : c<m.cols) (hk : k<rounds m.cols (blockSize m.cols)) (hl : lane<blockSize m.cols) :
    (pointer m r c).toNat=(m.key r c).address ∧
    (∀ byte,byte<4 → (pointer m r c).toNat+byte<m.view.storage.baseByte+m.view.storage.bytes ∧
      (pointer m r c).toNat+byte<2^64) ∧
    (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 lane).slt (BitVec.ofNat 64 m.cols)=
      decide (k*blockSize m.cols+lane<m.cols) ∧
    (BitVec.ofNat 64 (k*blockSize m.cols)+BitVec.ofNat 64 (blockSize m.cols)).toInt=
      (((k+1)*blockSize m.cols : Nat) : Int) := by
  have columns := Index.columns_bound m backed unit (by omega)
  exact ⟨(Index.active_pointer m metadata backed grid r c hr hc).1,
    (Index.active_pointer m metadata backed grid r c hr hc).2,
    Index.loop_mask_exact m.cols (blockSize m.cols) k lane (block_positive _) (block_bound _) columns hk hl,
    Index.loop_increment_exact m.cols (blockSize m.cols) k (block_positive _) (block_bound _) columns hk⟩

#axiomsClean block_profile
#axiomsClean prepared_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivTriton.Launch
