import bench.tritonbench_g.relu_strided_buffer.Proofs.Mathematical_WholeSource
import VeriTile.Triton.Float.Structural

/-!
Original installed signed source statements executed with opaque floating
values. Both element casts, the strict comparison and the real-zero cast are
kept in the resulting scalar expression. No floating law is assumed except
availability of the comparison implementation. This source is not a new tile:
it is the installed SourceSignedBranch.perTile embedded in ComputeKernel.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralTile
open VeriTile.Triton FP.Structural SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

abbrev Memory (α : Type) := RegionName → Nat → Cell α

/-- The only control flow in the installed source is one forRangeDyn. Its body
is made of assignments and a store. Expose that loop to the structural engine;
all other source statements are retained verbatim as alg statements. -/
theorem liftSourceStmt_projection (stmt : Stmt) :
    (liftSourceStmt stmt).toAlgorithm? = .ok stmt := by
  cases stmt <;> simp [liftSourceStmt, ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?_map_alg, bind, Except.bind]

theorem liftSourceBody_projection (body : List Stmt) :
    ComputeStmt.listToAlgorithm? (body.map liftSourceStmt) = .ok body := by
  induction body with
  | nil => rfl
  | cons stmt rest ih =>
    simp [ComputeStmt.listToAlgorithm?, liftSourceStmt_projection, ih, bind, Except.bind]

theorem perTileSource_projection (c : SignedViews) :
    (perTileSource c).toAlgorithm? =
      .ok (.mk [c.inputView.region] [c.outputView.region] c.perTile) := by
  simp [perTileSource]

theorem kernelSource_projection (c : SignedViews) (oneTile : Bool) (tiles : Nat) :
    (kernelSource c oneTile tiles).toAlgorithm? =
      .ok (.mk [c.inputView.region] [c.outputView.region]
        (SourceGridExecution.kernelBody c oneTile tiles)) := by
  simp [kernelSource, ComputeKernel.toAlgorithm?, liftSourceBody_projection, bind, Except.bind]

/-- Literal zero has precisely the source's real-to-FP32 conversion. -/
def scalarZero {α : Type} (M : Algebra α) : α :=
  M.cast none .real .fp32 (M.literal none .real 0)

/-- The two explicit source identity-dtype casts have not been erased. -/
def scalarResult {α : Type} (M : Algebra α) (cmp : α → α → Bool) (raw : α) : α :=
  let input := M.cast none .fp32 .fp32 raw
  M.cast none .fp32 .fp32 (if cmp (scalarZero M) input then input else scalarZero M)

def writeCell {α : Type} (mem : Memory α) (r : RegionName) (a : Nat) (v : Cell α) : Memory α :=
  fun r' a' => if r' = r ∧ a' = a then v else mem r' a'

def tileOutput {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (mem : Memory α) : Memory α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      writeCell current c.outputView.region (c.outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M cmp
          ((mem c.inputView.region (c.inputView.address (T * c.B + i.1.val))).read .fp32)))
    else current) mem

def rawValues {α : Type} [Inhabited α] (c : SignedViews) (T : Nat) (s : State α) :
    Values α .fp32 [c.B] :=
  fun i => if T * c.B + i.1.val < c.n then
    (s.mem c.inputView.region (c.inputView.address (T * c.B + i.1.val))).read .fp32
    else s.undef .fp32 c.inputView.region (c.inputView.address (T * c.B + i.1.val))

def inputValues {α : Type} [Inhabited α] (M : Algebra α) (c : SignedViews)
    (T : Nat) (s : State α) : Values α .fp32 [c.B] :=
  fun i => M.cast none .fp32 .fp32 (rawValues c T s i)

def selectedValues {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (s : State α) : Values α .fp32 [c.B] :=
  fun i => if cmp (scalarZero M) (inputValues M c T s i) then inputValues M c T s i else scalarZero M

def beforeStore {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (s : State α) : State α :=
  let s := s.setReg "tile_id0" .nat [] (fun _ => T)
  let s := s.setReg "offset0" .nat [] (fun _ => T * c.B)
  let s := s.setReg "in0_bptr" .nat [c.B] (fun i => c.inputView.address (T * c.B + i.1.val))
  let s := s.setReg "in0" .fp32 [c.B] (inputValues M c T s)
  let s := s.setReg "out0" .fp32 [c.B] (selectedValues M cmp c T s)
  s.setReg "out0_bptr" .nat [c.B] (fun i => c.outputView.address (T * c.B + i.1.val))

def sourceFinal {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (s : State α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      current.write c.outputView.region (c.outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M cmp (rawValues c T s i)))
    else current) (beforeStore M cmp c T s)

/-- Successful execution of the complete installed source perTile list. No
output state or scalar result is provided as a premise. -/
theorem perTile_exact {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : SignedViews) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M (c.perTile.map ComputeStmt.alg) s = some (sourceFinal M cmp c T s) := by
  simp [SignedViews.perTile, run, step, stepAlg, evalOp_unfold,
    dynamicAddress, dynamicIndex, dynamicMask, SignedTile.reluOp, SignedTile.zero32,
    numeric, numericLt, natLt, FP.Structural.bop, store, hc, ht,
    sourceFinal, beforeStore, rawValues,
    scalarResult, scalarZero, View.address, View.signedAddress, Region.cast, ofFloat, toFloat]
  rfl

theorem scatter_mem {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    (lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
      lanes.foldl (fun mem i => if active i then writeCell mem r (off i) (value i) else mem) s.mem := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi, if_true, State.write, writeCell] using
        ih (s.write r (off i) (value i))
    · simpa only [hi, if_false] using ih s

theorem sourceFinal_mem {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (c : SignedViews) (T : Nat) (s : State α) :
    (sourceFinal M cmp c T s).mem = tileOutput M cmp c T s.mem := by
  unfold sourceFinal tileOutput
  rw [scatter_mem]
  change _ = List.foldl _ s.mem _
  apply List.foldl_ext
  intro mem i _
  by_cases hi : T * c.B + i.1.val < c.n <;> simp [hi, rawValues]

theorem scatter_control {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (s : State α) :
    let final := lanes.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s
    final.regs = s.regs ∧ final.undef = s.undef ∧ final.pids = s.pids ∧ final.numPids = s.numPids := by
  induction lanes generalizing s with
  | nil => exact ⟨rfl, rfl, rfl, rfl⟩
  | cons i rest ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simpa only [hi, if_true, State.write] using ih (s.write r (off i) (value i))
    · simpa only [hi, if_false] using ih s

/-- The complete perTile contract consumed by source loop/grid composition.
Only the comparison's availability is assumed; scalar casts stay explicit. -/
theorem perTile_run {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32 = some cmp) (c : SignedViews) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    ∃ final, run M (c.perTile.map ComputeStmt.alg) s = some final ∧
      final.mem = tileOutput M cmp c T s.mem ∧ final.undef = s.undef ∧
      final.regs .nat [] "pid" = s.regs .nat [] "pid" ∧
      final.regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas" ∧
      final.pids = s.pids ∧ final.numPids = s.numPids := by
  have control := scatter_control c.outputView.region
    (fun i : TileIndex [c.B] => c.outputView.address (T * c.B + i.1.val))
    (fun i => Cell.mk .fp32 (scalarResult M cmp (rawValues c T s i)))
    (fun i => T * c.B + i.1.val < c.n) (TileShape.allIndices [c.B]) (beforeStore M cmp c T s)
  refine ⟨sourceFinal M cmp c T s, perTile_exact M cmp hc c T s ht,
    sourceFinal_mem M cmp c T s, ?_, ?_, ?_, ?_, ?_⟩
  · exact control.2.1
  · change (sourceFinal M cmp c T s).regs .nat [] "pid" = _
    rw [show (sourceFinal M cmp c T s).regs = (beforeStore M cmp c T s).regs from control.1]
    simp [beforeStore]
  · rw [show (sourceFinal M cmp c T s).regs = (beforeStore M cmp c T s).regs from control.1]
    simp [beforeStore]
  · exact control.2.2.1
  · exact control.2.2.2

private theorem fold_written {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (target : ι) (unique : ∀ i, active i → off i = off target → i = target)
    (lanes : List ι) (mem : Memory α) (h : mem r (off target) = value target) :
    (lanes.foldl (fun current i => if active i then writeCell current r (off i) (value i)
      else current) mem) r (off target) = value target := by
  induction lanes generalizing mem with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hi : active i
    · simp only [hi, if_true]
      by_cases ho : off i = off target
      · have hit := unique i hi ho
        subst i
        simp [writeCell]
      · simp [writeCell, Ne.symm ho, h]
    · simpa only [hi, if_false] using h

/-- Active-cell injectivity is sufficient; inactive negative-stride addresses
may collapse under toNat and are never stored. -/
theorem tileOutput_readback {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (c : SignedViews) (T : Nat) (mem : Memory α)
    {capacity : Nat} (valid : c.outputView.Valid c.n capacity)
    (strideNonzero : c.outputView.stride ≠ 0) (target : TileIndex [c.B])
    (active : T * c.B + target.1.val < c.n) :
    tileOutput M cmp c T mem c.outputView.region
      (c.outputView.address (T * c.B + target.1.val)) =
      .mk .fp32 (scalarResult M cmp
        ((mem c.inputView.region (c.inputView.address (T * c.B + target.1.val))).read .fp32)) := by
  have unique : ∀ i : TileIndex [c.B], T * c.B + i.1.val < c.n →
      c.outputView.address (T * c.B + i.1.val) =
        c.outputView.address (T * c.B + target.1.val) → i = target := by
    intro i hi ha
    have heq := address_injective c.outputView valid strideNonzero
      (a₁ := ⟨T * c.B + i.1.val, hi⟩) (a₂ := ⟨T * c.B + target.1.val, active⟩) ha
    apply Prod.ext
    · apply Fin.ext
      exact Nat.add_left_cancel (congrArg Fin.val heq)
    · exact Subsingleton.elim _ _
  obtain ⟨before, after, hall⟩ := List.append_of_mem (TileShape.mem_allIndices [c.B] target)
  unfold tileOutput
  rw [hall, List.foldl_append, List.foldl_cons]
  simp only [active, if_true]
  apply fold_written _ _ _ _ target unique
  simp [writeCell]

private theorem fold_frame {α ι : Type} (region : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (lanes : List ι) (mem : Memory α) (r : RegionName) (a : Nat)
    (outside : r ≠ region ∨ ∀ i, active i → a ≠ off i) :
    (lanes.foldl (fun current i => if active i then writeCell current region (off i) (value i)
      else current) mem) r a = mem r a := by
  induction lanes generalizing mem with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    by_cases hi : active i
    · simp only [hi, if_true, writeCell]
      rw [if_neg]
      rcases outside with hr | ha
      · exact fun h => hr h.1
      · exact fun h => ha i hi h.2
    · simp only [hi, if_false]

/-- Exact typed-cell frame, including inactive lanes and different regions. -/
theorem tileOutput_frame {α : Type} [Inhabited α] (M : Algebra α)
    (cmp : α → α → Bool) (c : SignedViews) (T : Nat) (mem : Memory α) (r : RegionName) (a : Nat)
    (outside : r ≠ c.outputView.region ∨ ∀ i : TileIndex [c.B],
      T * c.B + i.1.val < c.n → a ≠ c.outputView.address (T * c.B + i.1.val)) :
    tileOutput M cmp c T mem r a = mem r a := by
  exact fold_frame _ _ _ _ _ _ _ _ outside

#guard_msgs (drop info) in
#specNonCircular scalarResult avoiding [perTileSource, kernelSource, FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralTile
