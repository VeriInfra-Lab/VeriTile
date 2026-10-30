import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVNativeValues

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel KernelStaging BackwardQKVSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 10000000

theorem lower_dot (ops : Ops α) (literal : ℝ→α) (k : Nat) (a b : Fin k→Term α) :
    lower ops literal (.pendingDot k a b)=chain ops k
      (fun e=>if he : e<k then lower ops literal (a ⟨e,he⟩) else ops.zero)
      (fun e=>if he : e<k then lower ops literal (b ⟨e,he⟩) else ops.zero) ops.zero := rfl
theorem lower_materialized_dot (ops : Ops α) (literal : ℝ→α) (k : Nat) (a b : Fin k→Term α) :
    lower ops literal (.materialize (.pendingDot k a b))=chain ops k
      (fun e=>if he : e<k then lower ops literal (a ⟨e,he⟩) else ops.zero)
      (fun e=>if he : e<k then lower ops literal (b ⟨e,he⟩) else ops.zero) ops.zero := rfl

theorem score_gradient_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (i : ScoreIndex) :
    lower ops literal (scoreGradient (algebra ops.zero) c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 i)=
      Body.ds ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1) i.1.val i.2.1.val := by
  have hm : ∀ n b, lower ops literal (.binary .mul
      ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.2 i) b)=
      ops.mul (lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.2 i))
        (lower ops literal b) := by intro n b; cases n <;> rfl
  change ops.cast (lower ops literal (.binary .mul _ _))=_
  rw [hm,trajectory_score_local ops literal laws c s tag,score_decay ops literal laws]
  rfl

theorem partial_score_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (i : ScoreIndex) :
    lower ops literal (partialScore (algebra ops.zero) gradientDot c s scale i)=
      Body.partialScore ops c (xs ops literal c s) (literal scale) (s.pids 0) (s.pids 2) (s.pids 1) i.1.val i.2.1.val := by
  unfold partialScore
  rw [key_dot_plain_local ops c s tag]
  change ops.mul (lower ops literal (.pendingDot c.BK _ _))
    (lower ops literal (scoreDecay (algebra ops.zero) c s scale ⟨i.2.1,i.1,()⟩))=_
  rw [lower_dot,score_decay ops literal laws]
  congr 1
  apply chain_congr
  · intro e he
    simp only [he,↓reduceDIte]
    exact key_load ops literal laws c s ⟨i.1,⟨e,he⟩,()⟩
  · intro e he
    simp only [he,↓reduceDIte]
    exact query_load ops literal laws c s ⟨⟨e,he⟩,i.2.1,()⟩

theorem query_body_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (i : KeyIndex c) :
    lower ops literal (queryValue (algebra ops.zero) gradientDot c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 i)=
      Body.dq ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
        (history ops literal c s "h") i.1.val (feature c s i.2.1.val) := by
  have hm : ∀ n b, lower ops literal (.binary .mul
      ((trajectory (algebra ops.zero) gradientDot c s scale n).1.1 i) b)=
      ops.mul (lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.1 i))
        (lower ops literal b) := by intro n b; cases n <;> rfl
  change ops.cast (lower ops literal (.binary .add
    (.binary .mul ((trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1.1 i)
      (queryDecay (algebra ops.zero) c s scale ⟨i.1,()⟩))
    (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (scoreGradient (algebra ops.zero) c s scale
        (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1)
      (keyLoad (algebra ops.zero) c s) i)))=_
  erw [dot_plain (scoreGradient (algebra ops.zero) c s scale
    (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1)
    (keyLoad (algebra ops.zero) c s) i (fun e=>rfl)]
  change ops.cast (lower ops literal (.binary .add _ (.pendingDot 64 _ _)))=_
  rw [consume_dot,hm,trajectory_query_local ops literal laws c s tag,query_decay ops literal laws]
  congr 1
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact score_gradient_local ops literal laws c s tag scale ⟨i.1,⟨t,ht⟩,()⟩
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact key_load ops literal laws c s ⟨⟨t,ht⟩,i.2.1,()⟩

theorem key_body_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (i : KeyIndex c)
    (hc : s.pids 1<c.chunks) (hi : keyActive c s i) :
    lower ops literal (keyValue (algebra ops.zero) gradientDot c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 i)=
      Body.dk ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
        (history ops literal c s "dh") i.1.val (feature c s i.2.1.val) := by
  have hm : ∀ n b, lower ops literal (.binary .mul
      ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.1 i) b)=
      ops.mul (lower ops literal ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.1 i))
        (lower ops literal b) := by intro n b; cases n <;> rfl
  change ops.cast (lower ops literal (.binary .add
    (.binary .mul ((trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1.2.1 i)
      (keyDecay (algebra ops.zero) c s ⟨i.1,()⟩))
    (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (queryLoad (algebra ops.zero) c s)
      (scoreGradient (algebra ops.zero) c s scale
        (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1) ⟨i.2.1,i.1,()⟩)))=_
  rw [key_result_dot_local ops c s tag scale i hi]
  change ops.cast (lower ops literal (.binary .add _ (.materialize (.pendingDot 64 _ _))))=_
  rw [materialized_add,hm,trajectory_key_local ops literal laws c s tag,key_decay ops literal laws c s _ hc,
    lower_dot]
  congr 2
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact query_load ops literal laws c s ⟨i.2.1,⟨t,ht⟩,()⟩
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact score_gradient_local ops literal laws c s tag scale ⟨⟨t,ht⟩,i.1,()⟩

theorem value_carry_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (f : Nat) (i : ValueIndex c) :
    lower ops literal (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (keyLoad (algebra ops.zero) c s) (dhistoryLoad (algebra ops.zero) c s f) i)=
      chain ops c.BK
        (fun e=>kAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (s.pids 0*c.BK+e))
        (fun e=>masked ops c.K c.V (history ops literal c s "dh") (s.pids 0*c.BK+e) (column c f i.2.1.val)) ops.zero := by
  rw [key_dot_plain_local ops c s tag,lower_dot]
  apply chain_congr
  · intro e he
    simp only [he,↓reduceDIte]
    exact key_load ops literal laws c s ⟨i.1,⟨e,he⟩,()⟩
  · intro e he
    simp only [he,↓reduceDIte]
    exact dhistory_load ops literal laws c s f ⟨⟨e,he⟩,i.2.1,()⟩

theorem partial_body_local (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : ReadTags c s) (scale : ℝ) (f : Nat) (i : ValueIndex c)
    (hc : s.pids 1<c.chunks) :
    lower ops literal (partialValue (algebra ops.zero) gradientDot c s scale f i)=
      Body.dvPartial ops c (xs ops literal c s) (literal scale) (s.pids 0) (s.pids 2) (s.pids 1)
        (history ops literal c s "dh") i.1.val (column c f i.2.1.val) := by
  change ops.cast (lower ops literal (.binary .add
    (.binary .mul
      (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (keyLoad (algebra ops.zero) c s) (dhistoryLoad (algebra ops.zero) c s f) i)
      (keyDecay (algebra ops.zero) c s ⟨i.1,()⟩))
    (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (fun j=>.cast (partialScore (α := Term α) (algebra ops.zero) gradientDot c s scale j))
      (valueLoad (algebra ops.zero) c s "incoming" f) i)))=_
  erw [dot_plain (fun j=>.cast (partialScore (α := Term α) (algebra ops.zero) gradientDot c s scale j))
    (valueLoad (algebra ops.zero) c s "incoming" f) i (fun e=>rfl)]
  change ops.cast (lower ops literal (.binary .add _ (.pendingDot 64 _ _)))=_
  rw [consume_dot]
  have hseed : lower ops literal (.binary .mul
      (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (keyLoad (algebra ops.zero) c s) (dhistoryLoad (algebra ops.zero) c s f) i)
      (keyDecay (algebra ops.zero) c s ⟨i.1,()⟩))=
      ops.mul (lower ops literal (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (keyLoad (algebra ops.zero) c s) (dhistoryLoad (algebra ops.zero) c s f) i))
        (lower ops literal (keyDecay (algebra ops.zero) c s ⟨i.1,()⟩)) := by
    rw [key_dot_plain_local ops c s tag]
    rfl
  rw [hseed,value_carry_local ops literal laws c s tag,key_decay ops literal laws c s _ hc]
  congr 1
  apply chain_congr
  · intro t ht
    simp only [ht,↓reduceDIte]
    change ops.cast (lower ops literal (partialScore (algebra ops.zero) gradientDot c s scale ⟨i.1,⟨t,ht⟩,()⟩))=_
    rw [partial_score_local ops literal laws c s tag]
  · intro t ht
    simp only [ht,↓reduceDIte]
    exact incoming_load ops literal laws c s f ⟨⟨t,ht⟩,i.2.1,()⟩

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem score_gradient (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (i : ScoreIndex) :
    lower ops literal (scoreGradient (algebra ops.zero) c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 i)=
      Body.ds ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1) i.1.val i.2.1.val :=
  score_gradient_local ops literal laws c s (ReadTags.of_tagged c s tag) scale i

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem partial_score (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (i : ScoreIndex) :
    lower ops literal (partialScore (algebra ops.zero) gradientDot c s scale i)=
      Body.partialScore ops c (xs ops literal c s) (literal scale) (s.pids 0) (s.pids 2) (s.pids 1) i.1.val i.2.1.val :=
  partial_score_local ops literal laws c s (ReadTags.of_tagged c s tag) scale i

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem query_body (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (i : KeyIndex c) :
    lower ops literal (queryValue (algebra ops.zero) gradientDot c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 i)=
      Body.dq ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
        (history ops literal c s "h") i.1.val (feature c s i.2.1.val) :=
  query_body_local ops literal laws c s (ReadTags.of_tagged c s tag) scale i

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem key_body (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (i : KeyIndex c)
    (hc : s.pids 1<c.chunks) (hi : keyActive c s i) :
    lower ops literal (keyValue (algebra ops.zero) gradientDot c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 i)=
      Body.dk ops c (xs ops literal c s) (literal scale) (s.pids 2) (s.pids 1)
        (history ops literal c s "dh") i.1.val (feature c s i.2.1.val) :=
  key_body_local ops literal laws c s (ReadTags.of_tagged c s tag) scale i hc hi

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem value_carry (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (f : Nat) (i : ValueIndex c) :
    lower ops literal (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (keyLoad (algebra ops.zero) c s) (dhistoryLoad (algebra ops.zero) c s f) i)=
      chain ops c.BK
        (fun e=>kAt ops c (xs ops literal c s) (s.pids 2) (row s i.1.val) (s.pids 0*c.BK+e))
        (fun e=>masked ops c.K c.V (history ops literal c s "dh") (s.pids 0*c.BK+e) (column c f i.2.1.val)) ops.zero :=
  value_carry_local ops literal laws c s (ReadTags.of_tagged c s tag) f i

/-- Original global-provenance interface, retained verbatim as a corollary. -/
theorem partial_body (ops : Ops α) (literal : ℝ→α) (laws : LiteralLaws ops literal)
    (c : Shape) (s : State (Term α)) (tag : Tagged s) (scale : ℝ) (f : Nat) (i : ValueIndex c)
    (hc : s.pids 1<c.chunks) :
    lower ops literal (partialValue (algebra ops.zero) gradientDot c s scale f i)=
      Body.dvPartial ops c (xs ops literal c s) (literal scale) (s.pids 0) (s.pids 2) (s.pids 1)
        (history ops literal c s "dh") i.1.val (column c f i.2.1.val) :=
  partial_body_local ops literal laws c s (ReadTags.of_tagged c s tag) scale f i hc

#axiomsClean query_body_local
#axiomsClean key_body_local
#axiomsClean partial_body_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
