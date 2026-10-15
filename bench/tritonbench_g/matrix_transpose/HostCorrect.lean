import bench.tritonbench_g.matrix_transpose.Proofs.FPSoftware

/-! Logical composition of the original fp16-only Python wrapper. Arbitrary
sampled fp16 words model the result of randn; no property of RNG arithmetic is
needed to transpose them. Allocation supplies distinct logical regions.
This proves the single-program launch, shapes, strides, all output words and
full memory frame. Native allocation/ABI, Triton arange admissibility and the
compiler-to-device refinement remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.Host
open VeriTile.Triton FP.Structural FPBits
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

structure Input where
  input : RegionName
  output : RegionName
  m : Nat
  d : Nat

def parameters (h : Input) : Parameters :=
  ⟨h.input, h.output, max h.d 1, 1, max h.m 1, 1, h.m, h.d⟩

def count (h : Input) : Nat := h.m * h.d

def inputShape (h : Input) : List Nat := [h.m, h.d]
def outputShape (h : Input) : List Nat := [h.d, h.m]
def inputStrides (h : Input) : List Nat := [max h.d 1, 1]
def outputStrides (h : Input) : List Nat := [max h.m 1, 1]

/-- Fresh contiguous input/output allocations with arbitrary sampled input
payloads and the wrapper's exact zero-initialized output. No existing logical
memory outside the two allocated slices changes. -/
def allocate (h : Input) (initial : State Word) (samples : Nat → Word) : State Word :=
  { initial with mem := fun r o =>
      if r = h.input ∧ o < count h then .mk .fp16 (samples o)
      else if r = h.output ∧ o < count h then .mk .fp16 0
      else initial.mem r o }

/-- The original grid is (1,), with one fresh program register file. -/
def launchState (h : Input) (initial : State Word) (samples : Nat → Word) : State Word :=
  { allocate h initial samples with
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1 }

noncomputable def run (h : Input) (initial : State Word) (samples : Nat → Word) : Option (State Word) :=
  exec unusedPrimitives (source Bool.false (parameters h)) (launchState h initial samples)

/-- Flat row-major transpose specification, independent of source execution
and of the existing scatter witness. -/
def expectedMemory (h : Input) (initial : State Word) (samples : Nat → Word) :
    RegionName → Nat → Cell Word :=
  fun r o => if r = h.output ∧ o < count h
    then .mk .fp16 (samples ((o % h.m) * h.d + o / h.m))
    else (allocate h initial samples).mem r o

private theorem flat_lt (m d r c : Nat) (hr : r < m) (hc : c < d) :
    r * d + c < m * d := by
  have next := Nat.mul_le_mul_right d (Nat.succ_le_of_lt hr)
  rw [Nat.succ_mul] at next
  omega

theorem output_injective (h : Input) : Function.Injective (outputOffset (parameters h)) := by
  intro a b eq
  have ha := a.2.1.isLt
  have hb := b.2.1.isLt
  change a.2.1.val < h.m at ha
  change b.2.1.val < h.m at hb
  have hm : 1 ≤ h.m := by omega
  simp only [outputOffset, parameters, Nat.max_eq_left hm, Nat.mul_one] at eq
  have lower := congrArg (fun o => o % h.m) eq
  dsimp only at lower
  rw [Nat.mul_add_mod_of_lt ha, Nat.mul_add_mod_of_lt hb] at lower
  have upper : a.1.val = b.1.val :=
    Nat.mul_right_cancel (by omega : 0 < h.m) (by omega)
  apply Prod.ext (Fin.ext upper)
  exact Prod.ext (Fin.ext lower) (Subsingleton.elim _ _)

theorem output_in_bounds (h : Input) (i : TileIndex [h.d,h.m]) :
    outputOffset (parameters h) i < count h := by
  have hm : 1 ≤ h.m := by have := i.2.1.isLt; omega
  simpa [outputOffset, parameters, count, Nat.mul_comm, Nat.max_eq_left hm] using
    flat_lt h.d h.m i.1.val i.2.1.val i.1.isLt i.2.1.isLt

theorem input_in_bounds (h : Input) (i : TileIndex [h.d,h.m]) :
    inputOffset (parameters h) i < count h := by
  have hd : 1 ≤ h.d := by have := i.1.isLt; omega
  simpa [inputOffset, parameters, count, Nat.max_eq_left hd] using
    flat_lt h.m h.d i.2.1.val i.1.val i.2.1.isLt i.1.isLt

private theorem input_sample (h : Input) (initial : State Word) (samples : Nat → Word)
    (i : TileIndex [h.d,h.m]) :
    outputValue (parameters h) (launchState h initial samples) i =
      samples (i.2.1.val * h.d + i.1.val) := by
  have hd : 1 ≤ h.d := by have := i.1.isLt; omega
  have bound := input_in_bounds h i
  simp only [inputOffset, parameters, Nat.max_eq_left hd, Nat.mul_one] at bound
  simp [outputValue, inputOffset, parameters, Nat.max_eq_left hd, launchState, allocate, bound, Cell.read_same]

private theorem flat_coordinate (h : Input) (o : Nat) (ho : o < count h) :
    ∃ i : TileIndex [h.d,h.m], outputOffset (parameters h) i = o ∧
      i.1.val = o / h.m ∧ i.2.1.val = o % h.m := by
  have hm : 0 < h.m := by
    unfold count at ho
    by_contra nz
    have : h.m = 0 := by omega
    simp [this] at ho
  have high : o / h.m < h.d := by
    apply (Nat.div_lt_iff_lt_mul hm).2
    simpa [count, Nat.mul_comm] using ho
  let i : TileIndex [h.d,h.m] := ⟨⟨o / h.m, high⟩, ⟨o % h.m, Nat.mod_lt o hm⟩, ()⟩
  refine ⟨i, ?_, rfl, rfl⟩
  simpa [i, outputOffset, parameters, Nat.max_eq_left hm, Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div o h.m

theorem full_memory (h : Input) (initial : State Word) (samples : Nat → Word) :
    (execution (parameters h) (launchState h initial samples)).mem =
      expectedMemory h initial samples := by
  funext r o
  by_cases active : r = h.output ∧ o < count h
  · obtain ⟨i, offset, hi, hj⟩ := flat_coordinate h o active.2
    rcases active with ⟨rfl, ho⟩
    have read := readback (parameters h) (launchState h initial samples)
      (output_injective h) i
    rw [offset, input_sample h initial samples i, hi, hj] at read
    simpa [expectedMemory, ho] using read
  · have outside : r ≠ h.output ∨ ∀ i : TileIndex [h.d,h.m], o ≠ outputOffset (parameters h) i := by
      by_cases hr : r = h.output
      · right
        intro i eq
        have := output_in_bounds h i
        apply active
        exact ⟨hr, by simpa [eq] using this⟩
      · exact Or.inl hr
    have unchanged := frame (parameters h) (launchState h initial samples) r o outside
    simpa [expectedMemory, active, launchState] using unchanged

/-- The wrapper's fresh allocations and single original kernel launch produce
the exact sampled input transposed, preserving every fp16 bit pattern. The
injectivity premise of the kernel theorem is discharged by contiguous output
strides. This is a logical host proof, not a claim that every Nat dimension is
accepted by the native compiler. -/
specification matrix_transpose_host_correctness
    (h : Input) (initial : State Word) (samples : Nat → Word)
    (fresh : h.input ≠ h.output) :
    ∃ final, run h initial samples = some final ∧
      final.mem = expectedMemory h initial samples ∧
      (inputShape h).prod = (outputShape h).prod ∧
      (∀ r : Fin h.m, ∀ c : Fin h.d,
        final.mem h.output (c.val * h.m + r.val) =
          .mk .fp16 (samples (r.val * h.d + c.val))) ∧
      (∀ o, o < count h → final.mem h.input o = .mk .fp16 (samples o)) ∧
      (∀ region o, (region ≠ h.input ∨ count h ≤ o) →
        (region ≠ h.output ∨ count h ≤ o) → final.mem region o = initial.mem region o) := by
  refine ⟨execution (parameters h) (launchState h initial samples),
    execute unusedPrimitives Bool.false (parameters h) (launchState h initial samples),
    full_memory h initial samples, ?_, ?_, ?_, ?_⟩
  · simp [inputShape, outputShape, Nat.mul_comm]
  · intro r c
    have hm : 1 ≤ h.m := by have := r.isLt; omega
    let i : TileIndex [h.d,h.m] := ⟨c,r,()⟩
    have read := readback (parameters h) (launchState h initial samples) (output_injective h) i
    rw [input_sample h initial samples i] at read
    simpa [i, outputOffset, parameters, Nat.max_eq_left hm] using read
  · intro o ho
    rw [full_memory]
    simp [expectedMemory, fresh, allocate, ho]
  · intro region o hin hout
    have ni : ¬(region = h.input ∧ o < count h) := by
      intro both
      rcases hin with hn | hn
      · exact hn both.1
      · omega
    have no : ¬(region = h.output ∧ o < count h) := by
      intro both
      rcases hout with hn | hn
      · exact hn both.1
      · omega
    rw [full_memory]
    simp [expectedMemory, allocate, ni, no]

/-- Under a concrete signed32 element-offset bound, every logical input and
output address is representable. This arithmetic fact alone is not a native
pointer/ABI refinement theorem. -/
theorem signed32_offset_bounds (h : Input) (bound : count h ≤ 2 ^ 31)
    (i : TileIndex [h.d,h.m]) :
    inputOffset (parameters h) i < 2 ^ 31 ∧ outputOffset (parameters h) i < 2 ^ 31 :=
  ⟨(input_in_bounds h i).trans_le bound, (output_in_bounds h i).trans_le bound⟩

#axiomsClean output_injective
#axiomsClean full_memory
#axiomsClean matrix_transpose_host_correctness
#axiomsClean signed32_offset_bounds
#specNonCircular expectedMemory avoiding [source, execution, FP.Structural.exec, run]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose.Host
