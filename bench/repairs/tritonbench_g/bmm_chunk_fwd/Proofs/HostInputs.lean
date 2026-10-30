import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.HostLaunch

/-! Tensor-level, value-preserving host preparation interface. These are the
usual external Torch shape/stride/copy contracts, specialized to the source's
exact copy branch. The derivation below supplies every program's pins and
address bounds and does not assume any output value or successful execution. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host
open VeriTile.Triton
noncomputable section

def inputOffset (v : Strides) (b r g k : Nat) :=
  b * v.batch + r * v.row + g * v.head + k * v.column

structure TensorInputs (h : Call) (x : Inputs h) (s : BlockState) (bounds : RegionBounds) : Prop where
  groups : 0 < h.shape.groups
  freshA : h.A ≠ h.Out
  freshB : h.B ≠ h.Out
  freshSeq : Region.cast h.Seq ≠ h.Out
  aBounds : ∀ b r g (k : Fin h.shape.K), b < h.shape.batch → r < h.shape.seqlen → g < h.shape.groups →
    inputOffset (prepared h.shape h.aStrides) b r g k < bounds h.A
  bBounds : ∀ b r g (k : Fin h.shape.K), b < h.shape.batch → r < h.shape.seqlen → g < h.shape.groups →
    inputOffset (prepared h.shape h.bStrides) b r g k < bounds h.B
  seqBounds : h.hasSeq = Bool.true → ∀ b r, b < h.shape.batch → r < h.shape.seqlen →
    b * h.seqBatch + r * h.seqRow < bounds (Region.cast h.Seq)
  aPins : ∀ b r g (k : Fin h.shape.K), b < h.shape.batch → r < h.shape.seqlen → g < h.shape.groups →
    s.readMem h.A (inputOffset (prepared h.shape h.aStrides) b r g k) = x.a b r g k
  bPins : ∀ b r g (k : Fin h.shape.K), b < h.shape.batch → r < h.shape.seqlen → g < h.shape.groups →
    s.readMem h.B (inputOffset (prepared h.shape h.bStrides) b r g k) = x.b b r g k
  seqPins : h.hasSeq = Bool.true → ∀ b r, b < h.shape.batch → r < h.shape.seqlen →
    s.readMemValue .int (Region.cast h.Seq) (b * h.seqBatch + r * h.seqRow) = x.seq b r

theorem local_to_sequence (h : Call) (p r : Nat)
    (hr : r < RealViews.length h.parameters p) :
    p / h.shape.groups * h.shape.chunk + r < h.shape.seqlen := by
  change r < min h.shape.chunk (h.shape.seqlen - p / h.shape.groups * h.shape.chunk) at hr
  omega

theorem a_window (h : Call) (c : Config) (p₀ p₁ p₂ : Nat) (i : Fin c.M) (k : Fin h.shape.K) :
    (RealViews.io Bool.false c.M c.N c.K h.parameters).aWindow p₀ p₁ p₂ (i,k,PUnit.unit) =
      inputOffset (prepared h.shape h.aStrides) p₁
        (p₂ / h.shape.groups * h.shape.chunk + RealViews.row c.M c.N h.parameters p₀ i)
        (p₂ % h.shape.groups) k := by
  change (p₁ * (prepared h.shape h.aStrides).batch +
    p₂ / h.shape.groups * h.shape.chunk * (prepared h.shape h.aStrides).row +
    (p₂ - p₂ / h.shape.groups * h.shape.groups) * (prepared h.shape h.aStrides).head) +
    (RealViews.row c.M c.N h.parameters p₀ i * (prepared h.shape h.aStrides).row +
      k.val * (prepared h.shape h.aStrides).column) = _
  rw [← Nat.mod_eq_sub_div_mul]
  unfold inputOffset
  ring

theorem b_window (h : Call) (c : Config) (p₀ p₁ p₂ : Nat) (j : Fin c.N) (k : Fin h.shape.K) :
    (RealViews.io Bool.false c.M c.N c.K h.parameters).bWindow p₀ p₁ p₂ (k,j,PUnit.unit) =
      inputOffset (prepared h.shape h.bStrides) p₁
        (p₂ / h.shape.groups * h.shape.chunk + RealViews.column c.N h.parameters p₀ j)
        (p₂ % h.shape.groups) k := by
  change (p₁ * (prepared h.shape h.bStrides).batch +
    p₂ / h.shape.groups * h.shape.chunk * (prepared h.shape h.bStrides).row +
    (p₂ - p₂ / h.shape.groups * h.shape.groups) * (prepared h.shape h.bStrides).head) +
    (k.val * (prepared h.shape h.bStrides).column +
      RealViews.column c.N h.parameters p₀ j * (prepared h.shape h.bStrides).row) = _
  rw [← Nat.mod_eq_sub_div_mul]
  unfold inputOffset
  ring

theorem sequence_window (h : Call) (hs : h.hasSeq = Bool.true) (p₁ p₂ r : Nat) :
    p₁ * h.parameters.sBatch + RealViews.chunk h.parameters p₂ * h.parameters.chunk * h.parameters.sRow +
      r * h.parameters.sRow = p₁ * h.seqBatch + (p₂ / h.shape.groups * h.shape.chunk + r) * h.seqRow := by
  simp only [Call.parameters, RealViews.chunk, hs, ↓reduceIte]
  ring

theorem tensor_pins (h : Call) (c : Config) (x : Inputs h) (s : BlockState) (bounds : RegionBounds)
    (ht : TensorInputs h x s bounds) : Pinned h c x s := by
  intro p
  constructor
  · intro i k hi
    change s.readMem h.A ((RealViews.io Bool.false c.M c.N c.K h.parameters).aWindow
      (p0 p) (p1 p) (p2 p) (i,k,PUnit.unit)) = _
    rw [a_window]
    exact ht.aPins _ _ _ k (p1_lt p) (local_to_sequence h _ _ hi.2) (Nat.mod_lt _ ht.groups)
  · intro k j hj
    change s.readMem h.B ((RealViews.io Bool.false c.M c.N c.K h.parameters).bWindow
      (p0 p) (p1 p) (p2 p) (k,j,PUnit.unit)) = _
    rw [b_window]
    exact ht.bPins _ _ _ k (p1_lt p) (local_to_sequence h _ _ hj.2) (Nat.mod_lt _ ht.groups)
  · intro hs i hi
    change s.readMemValue .int (Region.cast h.Seq)
      (p1 p * h.parameters.sBatch + RealViews.chunk h.parameters (p2 p) * h.parameters.chunk * h.parameters.sRow +
        RealViews.row c.M c.N h.parameters (p0 p) i * h.parameters.sRow) = _
    rw [sequence_window h hs]
    exact ht.seqPins hs _ _ (p1_lt p) (local_to_sequence h _ _ hi.2)
  · intro hs j hj
    change s.readMemValue .int (Region.cast h.Seq)
      (p1 p * h.parameters.sBatch + RealViews.chunk h.parameters (p2 p) * h.parameters.chunk * h.parameters.sRow +
        RealViews.column c.N h.parameters (p0 p) j * h.parameters.sRow) = _
    rw [sequence_window h hs]
    exact ht.seqPins hs _ _ (p1_lt p) (local_to_sequence h _ _ hj.2)

theorem tensor_bounds (h : Call) (c : Config) (x : Inputs h) (s : BlockState) (bounds : RegionBounds)
    (ht : TensorInputs h x s bounds) (ho : bounds h.Out = h.shape.outputSize) (p : PID h c) :
    RealIO.Bounds (RealViews.io Bool.false c.M c.N c.K h.parameters) (s.withGridIndex p) bounds := by
  constructor
  · intro i hi
    change (RealViews.io Bool.false c.M c.N c.K h.parameters).aWindow
      (p0 p) (p1 p) (p2 p) i < _
    rw [show i = (i.1,i.2.1,PUnit.unit) from rfl, a_window]
    exact ht.aBounds _ _ _ i.2.1 (p1_lt p) (local_to_sequence h _ _ hi.2) (Nat.mod_lt _ ht.groups)
  · intro i hi
    change (RealViews.io Bool.false c.M c.N c.K h.parameters).bWindow
      (p0 p) (p1 p) (p2 p) i < _
    rw [show i = (i.1,i.2.1,PUnit.unit) from rfl, b_window]
    exact ht.bBounds _ _ _ i.1 (p1_lt p) (local_to_sequence h _ _ hi.2) (Nat.mod_lt _ ht.groups)
  · intro hs i hi
    change p1 p * h.parameters.sBatch + RealViews.chunk h.parameters (p2 p) * h.parameters.chunk * h.parameters.sRow +
      RealViews.row c.M c.N h.parameters (p0 p) i.1 * h.parameters.sRow < _
    rw [sequence_window h hs]
    exact ht.seqBounds hs _ _ (p1_lt p) (local_to_sequence h _ _ hi.2)
  · intro hs j hj
    change p1 p * h.parameters.sBatch + RealViews.chunk h.parameters (p2 p) * h.parameters.chunk * h.parameters.sRow +
      RealViews.column c.N h.parameters (p0 p) j.1 * h.parameters.sRow < _
    rw [sequence_window h hs]
    exact ht.seqBounds hs _ _ (p1_lt p) (local_to_sequence h _ _ hj.2)
  · intro i hi
    change offset h c p i < bounds h.Out
    rw [ho]
    exact output_bound h c p i hi

/-- Every other program may reuse the original logical tensors after any one
program executes. This proves read stability across the complete launch. -/
theorem tensors_stable (h : Call) (c : Config) (hc : c ∈ configurations)
    (swapped : Bool) (x : Inputs h) (s : BlockState) (bounds : RegionBounds)
    (ht : TensorInputs h x s bounds) (p : PID h c) :
    ∃ t, exec (RealProgram.kernel swapped c.M c.N c.K h.parameters).toAlgKernel
      (s.withGridIndex p) = some t ∧ TensorInputs h x t bounds := by
  obtain ⟨t, he, _, hf⟩ := program_executes h c hc swapped x s (tensor_pins h c x s bounds ht) p
  have hmem (r : RegionName) (hr : r ≠ h.Out) (a : Nat) : t.mem r a = s.mem r a :=
    hf r a (fun hw => hr hw.1)
  refine ⟨t, he, { ht with aPins := ?_, bPins := ?_, seqPins := ?_ }⟩
  · intro b r g k hb hr hg
    simpa only [BlockState.readMem, hmem _ ht.freshA] using ht.aPins b r g k hb hr hg
  · intro b r g k hb hr hg
    simpa only [BlockState.readMem, hmem _ ht.freshB] using ht.bPins b r g k hb hr hg
  · intro hs b r hb hr
    simpa only [BlockState.readMemValue, BlockState.readMemTyped, hmem _ ht.freshSeq]
      using ht.seqPins hs b r hb hr

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host

#guard_msgs (drop info) in
#auditModuleAxioms
