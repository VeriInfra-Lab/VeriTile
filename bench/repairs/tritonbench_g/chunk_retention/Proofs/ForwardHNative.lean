import bench.repairs.tritonbench_g.chunk_retention.Proofs.KernelNativeModel

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.H
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging ForwardHSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def xs (ops : Ops α) (literal : ℝ→α) (c : Shape) (seed : Bool) (s : State (Term α)) :=
  inputs ops literal c s.mem seed Bool.false

theorem block_decay (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (n : Nat) :
    lower ops literal (blockDecay (algebra ops.zero) c s n)=decay ops c (s.pids 2) n := by
  by_cases hn : n=64
  · subst n
    change ops.exp2 (ops.mul (literal 64)
      (lower ops literal (KernelModel.rate (algebra ops.zero) c (s.pids 2))))=_
    rw [laws.blockSize,KernelNative.rate_eq ops literal laws]
    rfl
  · simp only [blockDecay,hn,↓reduceIte]
    change ops.exp2 (ops.mul (ops.integer n)
      (lower ops literal (KernelModel.rate (algebra ops.zero) c (s.pids 2))))=_
    rw [KernelNative.rate_eq ops literal laws]
    rfl

theorem key_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (seed : Bool) (s : State (Term α)) (n : Nat) (i : TileIndex [c.BK,64]) :
    lower ops literal (keyLoad (algebra ops.zero) c s n i)=
      kAt ops c (xs ops literal c seed s) (s.pids 2) (n*64+i.2.1.val) (feature c s i.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]

theorem value_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (seed : Bool) (s : State (Term α)) (n : Nat) (i : TileIndex [64,c.BV]) :
    lower ops literal (valueLoad (algebra ops.zero) c s n i)=
      vAt ops c (xs ops literal c seed s) (s.pids 2) (n*64+i.1.val) (column c s i.2.1.val) := by
  simp [valueLoad,vAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]

theorem gated_value (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (seed : Bool) (s : State (Term α)) (n : Nat) (i : TileIndex [64,c.BV]) :
    lower ops literal (gatedValue (algebra ops.zero) c s n i)=
      ops.cast (ops.mul
        (vAt ops c (xs ops literal c seed s) (s.pids 2) (n*64+i.1.val) (column c s i.2.1.val))
        (decay ops c (s.pids 2) ((actualLength c n:Int)-i.1.val-1))) := by
  change ops.cast (ops.mul (lower ops literal (valueLoad (algebra ops.zero) c s n i))
    (lower ops literal (KernelModel.decay (algebra ops.zero) c (s.pids 2)
      ((actualLength c n:Int)-i.1.val-1))))=_
  rw [value_load ops literal laws c seed,KernelNative.decay_eq ops literal laws]

theorem initial_reference (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (c : Shape) (seed : Bool)
    (s : State (Term α)) (i : Index c) :
    lower ops literal (initValue (algebra ops.zero) c seed s i)=
      Reference.history ops c (xs ops literal c seed s) (s.pids 2) 0
        (feature c s i.1.val) (column c s i.2.1.val) := by
  cases seed <;> by_cases hi : active c s i <;>
    simp [initValue,Reference.history,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero,
      active,stateAddress,castZero] at hi ⊢

theorem update_body (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (seed : Bool) (s : State (Term α)) (n : Nat)
    (carry : Index c → Term α) (ref : Tile α) (i : Index c) (hn : n<c.chunks)
    (hi : lower ops literal (carry i)=ref (feature c s i.1.val) (column c s i.2.1.val)) :
    lower ops literal (ForwardHSource.update (algebra ops.zero) dot c s n carry i)=
      Body.nextH ops c (xs ops literal c seed s) (s.pids 2) n ref
        (feature c s i.1.val) (column c s i.2.1.val) := by
  unfold ForwardHSource.update
  change lower ops literal (.binary .add
    (.binary .mul (blockDecay (algebra ops.zero) c s (actualLength c n)) (carry i))
    (.pendingDot 64 (fun t => keyLoad (algebra ops.zero) c s n ⟨i.1,t,()⟩)
      (fun t => gatedValue (algebra ops.zero) c s n ⟨t,i.2.1,()⟩)))=_
  rw [consume_dot]
  change chain ops 64 _ _ (ops.mul
    (lower ops literal (blockDecay (algebra ops.zero) c s (actualLength c n)))
    (lower ops literal (carry i)))=_
  rw [block_decay ops literal laws,hi]
  have hl : actualLength c n=c.length n := (c.length_source n hn).symm
  rw [hl]
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact key_load ops literal laws c seed s n ⟨i.1,⟨t,ht⟩,()⟩
  · intro t ht
    simp only [ht,↓reduceDIte]
    rw [gated_value ops literal laws c seed,hl]

theorem trajectory_reference (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (c : Shape) (seed : Bool)
    (s : State (Term α)) (n : Nat) (hn : n≤c.chunks) (i : Index c) :
    lower ops literal ((trajectory (algebra ops.zero) dot c seed s n).1 i)=
      Reference.history ops c (xs ops literal c seed s) (s.pids 2) n
        (feature c s i.1.val) (column c s i.2.1.val) := by
  induction n with
  | zero => exact initial_reference ops literal laws castZero c seed s i
  | succ n ih =>
    change lower ops literal (ForwardHSource.update (algebra ops.zero) dot c s n
      (trajectory (algebra ops.zero) dot c seed s n).1 i)=_
    exact update_body ops literal laws c seed s n _
      (Reference.history ops c (xs ops literal c seed s) (s.pids 2) n) i (by omega) (ih (by omega))

theorem source_result (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (castZero : ops.cast ops.zero=ops.zero) (c : Shape) (seed final : Bool)
    (s : State (Term α)) :
    ∃ t, FP.Structural.exec (engine (withDot (algebra ops.zero) dot)) (kernel c seed final) s=some t ∧
      (∀ n i, n<c.chunks → active c s i →
        lower ops literal ((t.mem "h" (historyAddress c s n i)).read .real)=
          ops.cast (Reference.history ops c (xs ops literal c seed s) (s.pids 2) n
            (feature c s i.1.val) (column c s i.2.1.val))) ∧
      (final=Bool.true → ∀ i, active c s i →
        lower ops literal ((t.mem "ht" (stateAddress c s i)).read .real)=
          Reference.finalState ops c (xs ops literal c seed s) (s.pids 2)
            (feature c s i.1.val) (column c s i.2.1.val)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"h" ∨ ∀ n, n<c.chunks → ∀ i, active c s i → off≠historyAddress c s n i) →
        (final=Bool.false ∨ r≠"ht" ∨ ∀ i, active c s i → off≠stateAddress c s i) →
        t.mem r off=s.mem r off) := by
  obtain ⟨t,ht,hm⟩ := source_memory (algebra ops.zero) dot c seed final s
  refine ⟨t,ht,?_,?_,?_⟩
  · intro n i hn hi
    rw [hm,history_readback (algebra ops.zero) dot c seed final s n i hn hi]
    change ops.cast (lower ops literal ((trajectory (algebra ops.zero) dot c seed s n).1 i))=_
    rw [trajectory_reference ops literal laws castZero c seed s n (Nat.le_of_lt hn)]
  · intro hf i hi
    subst final
    rw [hm,final_readback (algebra ops.zero) dot c seed s i hi]
    change ops.cast (lower ops literal ((trajectory (algebra ops.zero) dot c seed s c.chunks).1 i))=_
    rw [trajectory_reference ops literal laws castZero c seed s c.chunks le_rfl]
    rfl
  · intro r off hh hf
    rw [hm]
    exact written_frame (algebra ops.zero) dot c seed final s r off hh hf

#axiomsClean trajectory_reference
#axiomsClean source_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.H
