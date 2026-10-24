import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVReadTags

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
open VeriTile Triton FP.Structural KernelModel KernelStaging BackwardQKVSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

def xs (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α)) :=
  inputs ops literal c s.mem Bool.false Bool.false
def history (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α))
    (r : RegionName) (e j : Nat) : α :=
  lower ops literal ((s.mem r (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+e*c.V+j)).read .real)
theorem query_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (i : TileIndex [c.BK,64]) :
    lower ops literal (queryLoad (algebra ops.zero) c s i)=
      qAt ops c (xs ops literal c s) (s.pids 2) (row s i.2.1.val) (feature c s i.1.val) := by
  simp [queryLoad,qAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem key_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (i : KeyIndex c) :
    lower ops literal (keyLoad (algebra ops.zero) c s i)=
      kAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (feature c s i.2.1.val) := by
  simp [keyLoad,kAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]
theorem value_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (f : Nat) (i : ValueIndex c) :
    lower ops literal (valueLoad (algebra ops.zero) c s "v" f i)=
      vAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (column c f i.2.1.val) := by
  simp [valueLoad,vAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]
theorem incoming_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (f : Nat) (i : ValueIndex c) :
    lower ops literal (valueLoad (algebra ops.zero) c s "incoming" f i)=
      gAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (column c f i.2.1.val) := by
  simp [valueLoad,gAt,masked,xs,inputs,algebra,lower_ite,lower,zero,laws.zero]
theorem history_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (f : Nat) (i : TileIndex [c.BV,c.BK]) :
    lower ops literal (historyLoad (algebra ops.zero) c s f i)=
      masked ops c.K c.V (history ops literal c s "h") (feature c s i.2.1.val) (column c f i.1.val) := by
  simp [historyLoad,history,masked,algebra,lower_ite,lower,zero,laws.zero,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,and_comm]
theorem dhistory_load (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (f : Nat) (i : TileIndex [c.BK,c.BV]) :
    lower ops literal (dhistoryLoad (algebra ops.zero) c s f i)=
      masked ops c.K c.V (history ops literal c s "dh") (feature c s i.1.val) (column c f i.2.1.val) := by
  simp [dhistoryLoad,history,masked,algebra,lower_ite,lower,zero,laws.zero]
theorem query_decay (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (scale : ℝ) (i : TileIndex [64]) :
    lower ops literal (queryDecay (algebra ops.zero) c s scale i)=
      ops.cast (ops.mul (decay ops c (s.pids 2) (i.1.val+1)) (literal scale)) := by
  change ops.cast (ops.mul (lower ops literal
    (KernelModel.decay (algebra ops.zero) c (s.pids 2) ((i.1.val:Int)+1))) (literal scale))=_
  rw [KernelNative.decay_eq ops literal laws]
theorem key_decay (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (i : TileIndex [64]) (hc : s.pids 1<c.chunks) :
    lower ops literal (keyDecay (algebra ops.zero) c s i)=
      decay ops c (s.pids 2) ((c.length (s.pids 1):Int)-i.1.val-1) := by
  unfold keyDecay
  rw [KernelNative.decay_eq ops literal laws,
    show ForwardHSource.actualLength c (s.pids 1)=c.length (s.pids 1) from (c.length_source _ hc).symm]
theorem score_decay (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (scale : ℝ) (i : ScoreIndex) :
    lower ops literal (scoreDecay (algebra ops.zero) c s scale i)=
      ops.mul (causalWeight ops c (s.pids 2) i.1.val i.2.1.val) (literal scale) := by
  unfold scoreDecay
  by_cases hi : i.2.1.val ≤ i.1.val
  · simp only [hi,↓reduceIte]
    change ops.mul (lower ops literal (KernelModel.decay (algebra ops.zero) c (s.pids 2)
      ((i.1.val:Int)-i.2.1.val))) (literal scale)=_
    rw [KernelNative.decay_eq ops literal laws]
    simp only [causalWeight,hi,↓reduceIte]
  · simp only [hi,↓reduceIte]
    change ops.mul (literal 0) (literal scale)=_
    rw [laws.zero]
    simp only [causalWeight,hi,↓reduceIte]

def queryReference (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α))
    (n : Nat) (i : KeyIndex c) : α :=
  blockChain ops c.BV
    (fun j => gAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) j)
    (fun j => masked ops c.K c.V (history ops literal c s "h") (feature c s i.2.1.val) j) n ops.zero
def keyReference (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α))
    (n : Nat) (i : KeyIndex c) : α :=
  blockChain ops c.BV
    (fun j => vAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) j)
    (fun j => masked ops c.K c.V (history ops literal c s "dh") (feature c s i.2.1.val) j) n ops.zero
def scoreReference (ops : Ops α) (literal : ℝ→α) (c : Shape) (s : State (Term α))
    (n : Nat) (i : ScoreIndex) : α :=
  blockChain ops c.BV
    (fun j => gAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) j)
    (fun j => vAt ops c (xs ops literal c s) (s.pids 2) (row s i.2.1.val) j) n ops.zero

theorem trajectory_query_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (n : Nat) (i : KeyIndex c) :
    lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.1 i)=
      queryReference ops literal c s n i := by
  induction n with
  | zero => exact laws.zero
  | succ n ih =>
    change lower ops literal (.binary .add ((trajectory (algebra ops.zero) gradientDot c s scale n).1.1 i)
      (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (valueLoad (algebra ops.zero) c s "incoming" n) (historyLoad (algebra ops.zero) c s n) i))=_
    rw [value_dot_plain_local ops c s tag "incoming" (by decide),consume_dot,ih]
    apply chain_congr
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact incoming_load ops literal laws c s n ⟨i.1,⟨j,hj⟩,()⟩
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact history_load ops literal laws c s n ⟨⟨j,hj⟩,i.2.1,()⟩
theorem trajectory_key_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (n : Nat) (i : KeyIndex c) :
    lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.1 i)=
      keyReference ops literal c s n i := by
  induction n with
  | zero => exact laws.zero
  | succ n ih =>
    change lower ops literal (.binary .add ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.1 i)
      (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (valueLoad (algebra ops.zero) c s "v" n)
        (fun (j : TileIndex [c.BV,c.BK]) => dhistoryLoad (α := Term α) (algebra ops.zero) c s n ⟨j.2.1,j.1,()⟩) i))=_
    rw [value_dot_plain_local ops c s tag "v" (by decide),consume_dot,ih]
    apply chain_congr
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact value_load ops literal laws c s n ⟨i.1,⟨j,hj⟩,()⟩
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact dhistory_load ops literal laws c s n ⟨i.2.1,⟨j,hj⟩,()⟩
theorem trajectory_score_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (n : Nat) (i : ScoreIndex) :
    lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.2 i)=
      scoreReference ops literal c s n i := by
  induction n with
  | zero => exact laws.zero
  | succ n ih =>
    change lower ops literal (.binary .add ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.2 i)
      (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (valueLoad (algebra ops.zero) c s "incoming" n)
        (fun (j : TileIndex [c.BV,64]) => valueLoad (α := Term α) (algebra ops.zero) c s "v" n ⟨j.2.1,j.1,()⟩) i))=_
    rw [value_dot_plain_local ops c s tag "incoming" (by decide),consume_dot,ih]
    apply chain_congr
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact incoming_load ops literal laws c s n ⟨i.1,⟨j,hj⟩,()⟩
    · intro j hj
      simp only [hj,↓reduceDIte]
      exact value_load ops literal laws c s n ⟨i.2.1,⟨j,hj⟩,()⟩

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem trajectory_query (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (n : Nat) (i : KeyIndex c) :
    lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.1 i)=
      queryReference ops literal c s n i :=
  trajectory_query_local ops literal laws c s (ReadTags.of_tagged c s tag) scale n i

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem trajectory_key (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (n : Nat) (i : KeyIndex c) :
    lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.1 i)=
      keyReference ops literal c s n i :=
  trajectory_key_local ops literal laws c s (ReadTags.of_tagged c s tag) scale n i

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem trajectory_score (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (n : Nat) (i : ScoreIndex) :
    lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.2 i)=
      scoreReference ops literal c s n i :=
  trajectory_score_local ops literal laws c s (ReadTags.of_tagged c s tag) scale n i

#axiomsClean trajectory_query_local
#axiomsClean trajectory_key_local
#axiomsClean trajectory_score_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
