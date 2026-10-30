import bench.tritonbench_g.chunk_retention.Sources.KernelNativeModel
import bench.tritonbench_g.chunk_retention.Sources.BackwardQKVReadback

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
open VeriTile Triton FP.Structural KernelModel KernelStaging BackwardQKVSource
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000

/-- Ordinary per-launch input-word provenance. This records the load operand
role and address, without assuming anything about a dot or its result. -/
def Tagged (s : State (Term α)) : Prop :=
  ∀ r off, ∃ x, (s.mem r off).read .real=Term.input r off x
def reify (s : State α) : State (Term α) where
  mem := fun r off => .mk .real (.input r off ((s.mem r off).read .real))
  regs := fun _ _ _ => none
  pids := s.pids
  numPids := s.numPids
  undef := fun d _ _ => defaultValue d
theorem reify_tagged (s : State α) : Tagged (reify s) := by
  intro r off
  exact ⟨(s.mem r off).read .real,rfl⟩

theorem dot_plain {m k n : Nat} (left : TileIndex [m,k] → Term α)
    (right : TileIndex [k,n] → Term α) (i : TileIndex [m,n])
    (h : ∀ e, fromInput "q" (left ⟨i.1,e,()⟩)=Bool.false) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[]) left right i=
      .pendingDot k (fun e=>left ⟨i.1,e,()⟩) (fun e=>right ⟨e,i.2.1,()⟩) := by
  simp [gradientDot,gradientDotValues,h,ofFloat,toFloat]
theorem dot_materialized {m k n : Nat} (left : TileIndex [m,k] → Term α)
    (right : TileIndex [k,n] → Term α) (i : TileIndex [m,n]) (hk : 0<k)
    (hl : fromInput "q" (left ⟨i.1,⟨0,hk⟩,()⟩)=Bool.true)
    (hr : weightedScore (right ⟨⟨0,hk⟩,i.2.1,()⟩)=Bool.true) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[]) left right i=
      .materialize (.pendingDot k (fun e=>left ⟨i.1,e,()⟩) (fun e=>right ⟨e,i.2.1,()⟩)) := by
  simp [gradientDot,gradientDotValues,hk,hl,hr,ofFloat,toFloat]

theorem key_tag (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : Tagged s) (i : KeyIndex c) :
    fromInput "q" (keyLoad (algebra ops.zero) c s i)=Bool.false := by
  unfold keyLoad
  simp only [algebra,fromInput]
  split_ifs
  · obtain ⟨x,hx⟩ := tag "k" (s.pids 2*(c.T*c.K)+row s i.1.val*c.K+feature c s i.2.1.val)
    rw [hx]
    rfl
  · rfl
theorem value_tag (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : Tagged s)
    (r : RegionName) (hr : r≠"q") (f : Nat) (i : ValueIndex c) :
    fromInput "q" (valueLoad (algebra ops.zero) c s r f i)=Bool.false := by
  unfold valueLoad
  simp only [algebra,fromInput]
  split_ifs
  · obtain ⟨x,hx⟩ := tag r (s.pids 2*(c.T*c.V)+row s i.1.val*c.V+column c f i.2.1.val)
    rw [hx]
    simp [fromInput,hr,Ne.symm hr]
  · rfl
theorem key_dot_plain (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : Tagged s)
    (right : TileIndex [c.BK,n] → Term α) (i : TileIndex [64,n]) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (keyLoad (algebra ops.zero) c s) right i=
      .pendingDot c.BK (fun e=>keyLoad (algebra ops.zero) c s ⟨i.1,e,()⟩) (fun e=>right ⟨e,i.2.1,()⟩) :=
  dot_plain _ _ i (fun e=>key_tag ops c s tag ⟨i.1,e,()⟩)
theorem value_dot_plain (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : Tagged s)
    (r : RegionName) (hr : r≠"q") (f : Nat) (right : TileIndex [c.BV,n] → Term α) (i : TileIndex [64,n]) :
    gradientDot.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (valueLoad (algebra ops.zero) c s r f) right i=
      .pendingDot c.BV (fun e=>valueLoad (algebra ops.zero) c s r f ⟨i.1,e,()⟩) (fun e=>right ⟨e,i.2.1,()⟩) :=
  dot_plain _ _ i (fun e=>value_tag ops c s tag r hr f ⟨i.1,e,()⟩)

theorem score_tag (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : Tagged s)
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
          (fun (j : TileIndex [c.BV,64]) => valueLoad (α := Term α) (algebra ops.zero) c s "v" n ⟨j.2.1,j.1,()⟩) i))
      (scoreDecay (algebra ops.zero) c s scale i)))=_
    rw [value_dot_plain ops c s tag "incoming" (by decide)]
    rfl

/-- Active q×dS is the transposed dK dot. Its locally derived tags force the
materialization boundary before the final add. -/
theorem key_result_dot (ops : Ops α) (c : Shape) (s : State (Term α)) (tag : Tagged s)
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
    obtain ⟨x,hx⟩ := tag "q" (s.pids 2*(c.T*c.K)+feature c s i.2.1.val+s.pids 1*64*c.K)
    simp [queryLoad,algebra,row,ht,hi.2,hx,fromInput]
  exact dot_materialized _ _ _ (by decide) htag (score_tag ops c s tag scale c.valueBlocks hn ⟨⟨0,by decide⟩,i.1,()⟩)

#specNonCircular Tagged avoiding [run,FP.Structural.exec,gradientDot]
#axiomsClean reify_tagged
#axiomsClean key_result_dot
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.QKV
