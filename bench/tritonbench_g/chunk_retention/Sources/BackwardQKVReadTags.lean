import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVNativeModel

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
open VeriTile Triton FP.Structural KernelModel KernelStaging BackwardQKVSource
noncomputable section
variable {α : Type} [Inhabited α]

/-- Provenance only at actual masked Q/K/V/incoming source reads. The finite
logical-input encoder can establish these facts; arbitrary unrelated memory
does not need to be globally tagged. No dot/result property is a field. -/
structure ReadTags (c : Shape) (s : State (Term α)) : Prop where
  key : ∀ i : KeyIndex c,keyActive c s i → ∃ x,
    (s.mem "k" (keyAddress c s i)).read .real=Term.input "k" (keyAddress c s i) x
  value : ∀ r : RegionName,r="incoming" ∨ r="v" → ∀ f (i : ValueIndex c),
    valueActive c s f i → ∃ x,
      (s.mem r (s.pids 2*(c.T*c.V)+row s i.1.val*c.V+column c f i.2.1.val)).read .real=
        Term.input r (s.pids 2*(c.T*c.V)+row s i.1.val*c.V+column c f i.2.1.val) x
  query : ∀ (e : Fin c.BK) (t : Fin 64),feature c s e.val<c.K → row s t.val<c.T → ∃ x,
    (s.mem "q" (s.pids 2*(c.T*c.K)+feature c s e.val+row s t.val*c.K)).read .real=
      Term.input "q" (s.pids 2*(c.T*c.K)+feature c s e.val+row s t.val*c.K) x

theorem ReadTags.of_tagged (c : Shape) (s : State (Term α)) (tag : Tagged s) : ReadTags c s where
  key := fun i _=>tag "k" (keyAddress c s i)
  value := fun r _ f i _=>tag r (s.pids 2*(c.T*c.V)+row s i.1.val*c.V+column c f i.2.1.val)
  query := fun e t _ _=>tag "q" (s.pids 2*(c.T*c.K)+feature c s e.val+row s t.val*c.K)

theorem key_tag_local (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : ReadTags c s)
    (i : KeyIndex c) : fromInput "q" (keyLoad (algebra ops.zero) c s i)=Bool.false := by
  unfold keyLoad
  simp only [algebra,fromInput]
  split_ifs with hi
  · obtain ⟨x,hx⟩ := tag.key i hi
    change (s.mem "k" (s.pids 2*(c.T*c.K)+row s i.1.val*c.K+feature c s i.2.1.val)).read .real=_ at hx
    rw [hx]
    rfl
  · rfl

theorem value_tag_local (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : ReadTags c s)
    (r : RegionName) (hr : r="incoming" ∨ r="v") (f : Nat) (i : ValueIndex c) :
    fromInput "q" (valueLoad (algebra ops.zero) c s r f i)=Bool.false := by
  unfold valueLoad
  simp only [algebra,fromInput]
  split_ifs with hi
  · obtain ⟨x,hx⟩ := tag.value r hr f i hi
    rw [hx]
    rcases hr with rfl|rfl <;> rfl
  · rfl

theorem key_dot_plain_local (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : ReadTags c s)
    (right : TileIndex [c.BK,n]→Term α) (i : TileIndex [64,n]) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (keyLoad (algebra ops.zero) c s) right i=
      .pendingDot c.BK (fun e=>keyLoad (algebra ops.zero) c s ⟨i.1,e,()⟩)
        (fun e=>right ⟨e,i.2.1,()⟩) :=
  dot_plain _ _ i (fun e=>key_tag_local ops c s tag ⟨i.1,e,()⟩)

theorem value_dot_plain_local (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : ReadTags c s)
    (r : RegionName) (hr : r="incoming" ∨ r="v") (f : Nat)
    (right : TileIndex [c.BV,n]→Term α) (i : TileIndex [64,n]) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (valueLoad (algebra ops.zero) c s r f) right i=
      .pendingDot c.BV (fun e=>valueLoad (algebra ops.zero) c s r f ⟨i.1,e,()⟩)
        (fun e=>right ⟨e,i.2.1,()⟩) :=
  dot_plain _ _ i (fun e=>value_tag_local ops c s tag r hr f ⟨i.1,e,()⟩)

theorem score_tag_local (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : ReadTags c s)
    (scale : ℝ) (n : Nat) (hn : 0<n) (i : ScoreIndex) :
    weightedScore (scoreGradient (algebra ops.zero) c s scale
      (trajectory (algebra ops.zero) gradientDot c s scale n).1 i)=Bool.true := by
  cases n with
  | zero => omega
  | succ n =>
    change weightedScore (.cast (.binary .mul
      (.binary .add ((trajectory (algebra ops.zero) gradientDot c s scale n).1.2.2 i)
        (gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
          (valueLoad (algebra ops.zero) c s "incoming" n)
          (fun (j : TileIndex [c.BV,64])=>valueLoad (α:=Term α) (algebra ops.zero) c s "v" n ⟨j.2.1,j.1,()⟩) i))
      (scoreDecay (algebra ops.zero) c s scale i)))=_
    rw [value_dot_plain_local ops c s tag "incoming" (by decide)]
    rfl

theorem key_result_dot_local (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : ReadTags c s)
    (scale : ℝ) (i : KeyIndex c) (hi : keyActive c s i) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (queryLoad (algebra ops.zero) c s)
      (scoreGradient (algebra ops.zero) c s scale (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1)
      ⟨i.2.1,i.1,()⟩=
      .materialize (.pendingDot 64 (fun e=>queryLoad (algebra ops.zero) c s ⟨i.2.1,e,()⟩)
        (fun e=>scoreGradient (algebra ops.zero) c s scale
          (trajectory (algebra ops.zero) gradientDot c s scale c.valueBlocks).1 ⟨e,i.1,()⟩)) := by
  have hn : 0<c.valueBlocks := by
    have h:=c.value_cover
    have hv:=c.valuePositive
    by_contra hz
    have he : c.valueBlocks=0 := by omega
    change c.V≤c.valueBlocks*c.BV at h
    rw [he] at h
    simp at h
    omega
  have htag : fromInput "q" (queryLoad (algebra ops.zero) c s ⟨i.2.1,⟨0,by decide⟩,()⟩)=Bool.true := by
    have ht : s.pids 1*64<c.T := by have h:=hi.1; dsimp [row] at h; omega
    obtain ⟨x,hx⟩ := tag.query i.2.1 ⟨0,by decide⟩ hi.2 (by simpa [row] using ht)
    simp only [row,Nat.add_zero] at hx
    simp [queryLoad,algebra,row,ht,hi.2,hx,fromInput]
  exact dot_materialized _ _ _ (by decide) htag
    (score_tag_local ops c s tag scale c.valueBlocks hn ⟨⟨0,by decide⟩,i.1,()⟩)

#specNonCircular ReadTags avoiding [run,FP.Structural.exec,gradientDot]
#axiomsClean ReadTags.of_tagged
#axiomsClean key_result_dot_local
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
