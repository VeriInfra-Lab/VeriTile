import bench.tritonbench_g.pow_scalar_tensor.Proofs.Mathematical_WholeSource

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeTile
open VeriTile.Triton FP.Structural WholeSource
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer
open SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

abbrev Memory (α : Type) := RegionName → Nat → Cell α

/-- Exactly the explicit scalar-base FP32 cast. -/
def scalarBase {α : Type} (M : Algebra α) (base : ℝ) : α :=
  M.cast (some .fp32) .real .real (M.literal (some .fp32) .real base)

/-- Actual input-element cast, libdevice FP32 power, and output-element cast.
No floating law or source-level output summary is assumed. -/
def scalarResult {α : Type} (M : Algebra α) (base : ℝ) (raw : α) : α :=
  M.cast none .real .fp32 (M.binary (some .fp32) .real .pow
    (scalarBase M base) (M.cast (some .fp32) .fp32 .real raw))

def writeCell {α : Type} (mem : Memory α) (r : RegionName) (a : Nat) (v : Cell α) : Memory α :=
  fun r' a' => if r' = r ∧ a' = a then v else mem r' a'

def tileOutput {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (mem : Memory α) : Memory α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      writeCell current c.outputView.region (c.outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M c.base
          ((mem c.inputView.region (c.inputView.address (T * c.B + i.1.val))).read .fp32)))
    else current) mem

def rawValues {α : Type} [Inhabited α] (c : Config) (T : Nat) (s : State α) :
    Values α .fp32 [c.B] :=
  fun i => if T * c.B + i.1.val < c.n then
    (s.mem c.inputView.region (c.inputView.address (T * c.B + i.1.val))).read .fp32
    else s.undef .fp32 c.inputView.region (c.inputView.address (T * c.B + i.1.val))

def inputValues {α : Type} [Inhabited α] (M : Algebra α) (c : Config)
    (T : Nat) (s : State α) : Values α .real [c.B] :=
  fun i => M.cast (some .fp32) .fp32 .real (rawValues c T s i)

def selectedValues {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (s : State α) : Values α .real [c.B] :=
  fun i => M.binary (some .fp32) .real .pow (scalarBase M c.base) (inputValues M c T s i)

def beforeStore {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (s : State α) : State α :=
  let s := s.setReg "tile_id0" .nat [] (fun _ => T)
  let s := s.setReg "offset0" .nat [] (fun _ => if c.candidate then c.B * T else T * c.B)
  let s := s.setReg "in0_bptr" .nat [c.B] (fun i => c.inputView.address (T * c.B + i.1.val))
  let s := s.setReg "in0" .real [c.B] (inputValues M c T s)
  let s := s.setReg "out0" .real [c.B] (selectedValues M c T s)
  s.setReg "out0_bptr" .nat [c.B] (fun i => c.outputView.address (T * c.B + i.1.val))

def sourceFinal {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (s : State α) : State α :=
  (TileShape.allIndices [c.B]).foldl (fun current i =>
    if T * c.B + i.1.val < c.n then
      current.write c.outputView.region (c.outputView.address (T * c.B + i.1.val))
        (.mk .fp32 (scalarResult M c.base (rawValues c T s i)))
    else current) (beforeStore M c T s)

/-- Successful execution of the complete installed source perTile list. No
output state or scalar result is provided as a premise. -/
theorem perTile_exact {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    run M c.perTile s = some (sourceFinal M c T s) := by
  cases hc : c.candidate <;>
    simp [Config.perTile, offset, hc, run, step, stepAlg, evalExpr, evalComputeOp,
      evalOp_unfold, dynamicAddress, dynamicIndex, dynamicMask,
      numeric, natLt, FP.Structural.bop, store, ht, sourceFinal, beforeStore, rawValues,
      inputValues, selectedValues, scalarResult, scalarBase, View.address,
      View.signedAddress, Region.cast, ofFloat, toFloat, Nat.mul_comm]
  all_goals try rfl
  have hm : (FP.Structural.bop (fun x y : Nat => x * y) Broadcast.nil
      (fun _ => c.B) (fun _ => T)) = (fun _ => T * c.B) := by
    funext i
    exact Nat.mul_comm c.B T
  rw [hm]
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

theorem sourceFinal_mem {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (s : State α) :
    (sourceFinal M c T s).mem = tileOutput M c T s.mem := by
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
The power primitive is unchanged and opaque; all three scalar casts stay explicit. -/
theorem perTile_run {α : Type} [Inhabited α] (M : Algebra α) (c : Config) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id" = some (fun _ => T)) :
    ∃ final, run M c.perTile s = some final ∧
      final.mem = tileOutput M c T s.mem ∧ final.undef = s.undef ∧
      final.regs .nat [] "pid" = s.regs .nat [] "pid" ∧
      final.regs .nat [] "num_ctas" = s.regs .nat [] "num_ctas" ∧
      final.pids = s.pids ∧ final.numPids = s.numPids := by
  have control := scatter_control c.outputView.region
    (fun i : TileIndex [c.B] => c.outputView.address (T * c.B + i.1.val))
    (fun i => Cell.mk .fp32 (scalarResult M c.base (rawValues c T s i)))
    (fun i => T * c.B + i.1.val < c.n) (TileShape.allIndices [c.B]) (beforeStore M c T s)
  refine ⟨sourceFinal M c T s, perTile_exact M c T s ht,
    sourceFinal_mem M c T s, ?_, ?_, ?_, ?_, ?_⟩
  · exact control.2.1
  · change (sourceFinal M c T s).regs .nat [] "pid" = _
    rw [show (sourceFinal M c T s).regs = (beforeStore M c T s).regs from control.1]
    simp [beforeStore]
  · rw [show (sourceFinal M c T s).regs = (beforeStore M c T s).regs from control.1]
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
    (c : Config) (T : Nat) (mem : Memory α)
    {capacity : Nat} (valid : c.outputView.Valid c.n capacity)
    (strideNonzero : c.outputView.stride ≠ 0) (target : TileIndex [c.B])
    (active : T * c.B + target.1.val < c.n) :
    tileOutput M c T mem c.outputView.region
      (c.outputView.address (T * c.B + target.1.val)) =
      .mk .fp32 (scalarResult M c.base
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
    (c : Config) (T : Nat) (mem : Memory α) (r : RegionName) (a : Nat)
    (outside : r ≠ c.outputView.region ∨ ∀ i : TileIndex [c.B],
      T * c.B + i.1.val < c.n → a ≠ c.outputView.address (T * c.B + i.1.val)) :
    tileOutput M c T mem r a = mem r a := by
  exact fold_frame _ _ _ _ _ _ _ _ outside

#guard_msgs (drop info) in
#specNonCircular scalarResult avoiding [WholeSource.kernel, WholeSource.runHost, FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.WholeTile
