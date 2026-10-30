import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardHComplete
import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHNative

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.BH
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging BackwardHSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def xs (ops : Ops α) (literal : ℝ→α) (c : Shape) (terminal : Bool) (s : State (Term α)) :=
  inputs ops literal c s.mem Bool.false terminal
theorem query_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (terminal : Bool) (s : State (Term α)) (n : Nat) (i : TileIndex [c.BK,64]) :
    lower ops literal (queryLoad (algebra ops.zero) c s n i)=
      qAt ops c (xs ops literal c terminal s) (s.pids 2) (chunk c n*64+i.2.1.val) (feature c s i.1.val) := by
  simp [queryLoad,qAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem incoming_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (terminal : Bool) (s : State (Term α)) (n : Nat) (i : TileIndex [64,c.BV]) :
    lower ops literal (incomingLoad (algebra ops.zero) c s n i)=
      gAt ops c (xs ops literal c terminal s) (s.pids 2) (chunk c n*64+i.1.val) (column c s i.2.1.val) := by
  simp [incomingLoad,gAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]
theorem gated_incoming (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (terminal : Bool) (s : State (Term α)) (scale : ℝ) (n : Nat) (i : TileIndex [64,c.BV]) :
    lower ops literal (gatedIncoming (algebra ops.zero) c s scale n i)=
      ops.cast (ops.mul (ops.mul
        (gAt ops c (xs ops literal c terminal s) (s.pids 2) (chunk c n*64+i.1.val) (column c s i.2.1.val))
        (decay ops c (s.pids 2) (i.1.val+1))) (literal scale)) := by
  change ops.cast (ops.mul (ops.mul (lower ops literal (incomingLoad (algebra ops.zero) c s n i))
    (lower ops literal (KernelModel.decay (algebra ops.zero) c (s.pids 2) ((i.1.val:Int)+1)))) (literal scale))=_
  rw [incoming_load ops literal laws c terminal,KernelNative.decay_eq ops literal laws]
theorem initial_reference (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (c : Shape) (terminal : Bool)
    (s : State (Term α)) (scale : ℝ) (i : Index c) :
    lower ops literal (initValue (algebra ops.zero) c terminal s i)=
      Reference.reverseCarry ops c (xs ops literal c terminal s) (literal scale) (s.pids 2) 0
        (feature c s i.1.val) (column c s i.2.1.val) := by
  cases terminal <;> by_cases hi : active c s i <;>
    simp [initValue,Reference.reverseCarry,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero,
      active,stateAddress,ForwardHSource.active,ForwardHSource.stateAddress,castZero] at hi ⊢

theorem update_body (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (terminal : Bool) (s : State (Term α)) (scale : ℝ) (n : Nat)
    (carry : Index c → Term α) (ref : Tile α) (i : Index c) (hn : n<c.chunks)
    (hi : lower ops literal (carry i)=ref (feature c s i.1.val) (column c s i.2.1.val)) :
    lower ops literal (BackwardHSource.update (algebra ops.zero) dot c s scale n carry i)=
      Body.nextDH ops c (xs ops literal c terminal s) (literal scale) (s.pids 2) (chunk c n) ref
        (feature c s i.1.val) (column c s i.2.1.val) := by
  unfold BackwardHSource.update
  change lower ops literal (.binary .add
    (.binary .mul (blockDecay (algebra ops.zero) c s (actualLength c (chunk c n))) (carry i))
    (.pendingDot 64 (fun t => queryLoad (algebra ops.zero) c s n ⟨i.1,t,()⟩)
      (fun t => gatedIncoming (algebra ops.zero) c s scale n ⟨t,i.2.1,()⟩)))=_
  rw [consume_dot]
  change chain ops 64 _ _ (ops.mul
    (lower ops literal (ForwardHSource.blockDecay (algebra ops.zero) c s (actualLength c (chunk c n))))
    (lower ops literal (carry i)))=_
  rw [H.block_decay ops literal laws,hi]
  have hc : chunk c n<c.chunks := by dsimp [chunk]; omega
  have hl : actualLength c (chunk c n)=c.length (chunk c n) := (c.length_source _ hc).symm
  rw [hl]
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact query_load ops literal laws c terminal s n ⟨i.1,⟨t,ht⟩,()⟩
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact gated_incoming ops literal laws c terminal s scale n ⟨⟨t,ht⟩,i.2.1,()⟩

theorem trajectory_reference (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (c : Shape) (terminal : Bool)
    (s : State (Term α)) (scale : ℝ) (n : Nat) (hn : n≤c.chunks) (i : Index c) :
    lower ops literal ((trajectory (algebra ops.zero) dot c s scale terminal n).1 i)=
      Reference.reverseCarry ops c (xs ops literal c terminal s) (literal scale) (s.pids 2) n
        (feature c s i.1.val) (column c s i.2.1.val) := by
  induction n with
  | zero => exact initial_reference ops literal laws castZero c terminal s scale i
  | succ n ih =>
    change lower ops literal (BackwardHSource.update (algebra ops.zero) dot c s scale n
      (trajectory (algebra ops.zero) dot c s scale terminal n).1 i)=_
    exact update_body ops literal laws c terminal s scale n _
      (Reference.reverseCarry ops c (xs ops literal c terminal s) (literal scale) (s.pids 2) n)
      i (by omega) (ih (by omega))

theorem source_result (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (c : Shape) (terminal : Bool)
    (s : State (Term α)) (scale : ℝ) :
    ∃ t, FP.Structural.exec (engine (withDot (algebra ops.zero) dot)) (kernel c scale terminal) s=some t ∧
      (∀ n i, n<c.chunks → active c s i →
        lower ops literal ((t.mem "dh" (historyAddress c s (chunk c n) i)).read .real)=
          ops.cast (Reference.reverseCarry ops c (xs ops literal c terminal s) (literal scale) (s.pids 2) n
            (feature c s i.1.val) (column c s i.2.1.val))) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"dh" ∨ ∀ n, n<c.chunks → ∀ i, active c s i → off≠historyAddress c s n i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory (algebra ops.zero) dot c scale terminal s
  refine ⟨t,ht,?_,?_⟩
  · intro n i hn hi
    rw [hm,history_readback (algebra ops.zero) dot c s scale terminal n i hn hi]
    change ops.cast (lower ops literal ((trajectory (algebra ops.zero) dot c s scale terminal n).1 i))=_
    rw [trajectory_reference ops literal laws castZero c terminal s scale n (Nat.le_of_lt hn)]
  · intro r off hf
    rw [hm]
    exact frame (algebra ops.zero) dot c s scale terminal r off hf

#axiomsClean trajectory_reference
#axiomsClean source_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.BH
