import bench.repairs.tritonbench_g.rotary_transform.Proofs.Guards

namespace VeriTile.Bench.Repairs.RotaryTransform.BlockSizeDefect

/-- The first even rotary width whose two-dimensional pointer tensor exceeds
Triton 3.7.1's frontend limit in each pairing branch. This is a source-shape
failure, independent of payload values and floating-point instruction choices. -/
def width (interleaved : Bool) : Nat := if interleaved then 262146 else 524290

def shape (interleaved : Bool) : Shape where
  batches := 1
  maximum := 1
  total := 1
  heads := 1
  width := width interleaved
  half := width interleaved / 2
  coefficientRows := 1
  varlen := false
  offsetTensor := false
  scalarOffset := 0
  interleaved := interleaved
  conjugate := false
  inplace := false

def tensor (identity base : Nat) (dimensions strides : List Nat) : Checked.Tensor where
  object := identity
  device := 0
  shape := dimensions
  strides := strides
  base := base
  allocationBase := base
  allocationBytes := 4 * dimensions.prod
  storageOffset := 0
  dtype := "float32"
  elementBytes := 4
  strided := true
  cuda := true
  contiguous := true
  negative := false
  conjugate := false

/-- Concrete separate, aligned, contained FP32 allocations, with canonical
singleton strides. No metadata Tensor is used by these fixed/scalar calls. -/
def call (interleaved : Bool) : Checked.Call where
  host := {
    shape := shape interleaved
    abi := ⟨false, false, 1⟩
    inputIdentity := 0
    freshIdentity := 1
    device := 0 }
  views := ⟨width interleaved / 2, 1, width interleaved / 2, 1, 1⟩
  input := tensor 0 16777216 [1, 1, 1, width interleaved]
    [width interleaved, width interleaved, width interleaved, 1]
  cosine := tensor 2 33554432 [1, width interleaved / 2] [width interleaved / 2, 1]
  sine := tensor 3 50331648 [1, width interleaved / 2] [width interleaved / 2, 1]
  cumulative := tensor 4 67108864 [0] [1]
  offsets := tensor 5 83886080 [0] [1]

def initial : Physical.Bytes := fun _ _ => 0

/-- The existing complete checked contract admits both counterexamples. -/
theorem checked_admission (interleaved : Bool) :
    Checked.check (call interleaved) initial = true := by
  cases interleaved <;> decide

/-- Fixed sequences with scalar offsets never inspect metadata bytes, so
admission is independent of all payload words, including the replay's ones. -/
theorem checked_admission_all_payloads (interleaved : Bool) (m : Physical.Bytes) :
    Checked.check (call interleaved) m = true := by
  have configuration_eq : Checked.configuration (call interleaved) m =
      Checked.configuration (call interleaved) initial := by
    simp only [Checked.configuration, Physical.callerConfiguration, call, shape,
      Physical.canonicalMetadata, Bool.false_eq_true, ↓reduceIte]
  have accepted := checked_admission interleaved
  unfold Checked.check at accepted ⊢
  rw [configuration_eq]
  exact accepted

noncomputable def realInputs : Memory ℝ := fun region _ => if region = .sine then 0 else 1

/-- The independent rotation target is well defined at every column of the
counterexample: a=b=cosine=1 and sine=0 give output 1 in either pairing mode. -/
theorem independent_target_one (interleaved : Bool) (column : Nat) :
    realFormula (Checked.configuration (call interleaved) initial) realInputs
      0 0 0 column = 1 := by
  simp [realFormula, realInputs, Checked.configuration, Physical.callerConfiguration,
    Config.shift, shape, call]

theorem valid_configuration (interleaved : Bool) :
    (Checked.configuration (call interleaved) initial).Valid :=
  Checked.config_valid _ _ (checked_admission interleaved)

theorem checked_integer_bounds (interleaved : Bool) :
    IndexABI (Checked.configuration (call interleaved) initial) :=
  Checked.indices _ _ (checked_admission interleaved)

theorem nonempty_launch (interleaved : Bool) :
    Host.empty (shape interleaved) = false ∧
    (shape interleaved).rowBlocks = 1 ∧
    (shape interleaved).blockM = 4 := by
  cases interleaved <;> decide

theorem chosen_blockK :
    (shape true).blockK = 524288 ∧ (shape false).blockK = 1048576 := by decide

/-- Shape of the first matrix-valued pointer expression in the selected
branch: rm[:,None] + rk[None,:], or rm[:,None] + rk_half[None,:]. -/
def pointerElements (s : Shape) : Nat :=
  s.blockM * (if s.interleaved then s.blockK else s.blockK / 2)

theorem oversized_pointer_tensor (interleaved : Bool) :
    pointerElements (shape interleaved) = 2097152 ∧
    1048576 < pointerElements (shape interleaved) := by
  cases interleaved <;> decide

/-- The mathematical/ABI contract alone does not imply the compiler's
necessary block-tensor-size condition. The actual compiler rejection is
replayed separately; this theorem does not model or assume compiler execution. -/
theorem admission_does_not_bound_source_block :
    ¬ (∀ (c : Checked.Call) (m : Physical.Bytes), Checked.check c m = true →
      pointerElements c.host.shape ≤ 1048576) := by
  intro bounded
  have bound := bounded (call true) initial (checked_admission true)
  have large := (oversized_pointer_tensor true).2
  change pointerElements (shape true) ≤ 1048576 at bound
  omega

/-- At the immediately preceding even widths, this particular frontend
size limit is met. This says nothing about backend resource availability. -/
theorem preceding_even_widths :
    pointerElements { shape true with width := 262144, half := 131072 } = 1048576 ∧
    pointerElements { shape false with width := 524288, half := 262144 } = 1048576 := by
  decide

#axiomsClean checked_admission
#axiomsClean checked_admission_all_payloads
#axiomsClean independent_target_one
#axiomsClean valid_configuration
#axiomsClean checked_integer_bounds
#axiomsClean nonempty_launch
#axiomsClean chosen_blockK
#axiomsClean oversized_pointer_tensor
#axiomsClean admission_does_not_bound_source_block
#axiomsClean preceding_even_widths
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RotaryTransform.BlockSizeDefect
