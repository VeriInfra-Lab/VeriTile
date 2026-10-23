import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.KernelExecution
import bench.tritonbench_g.quant_transpose_kernel.Proofs.GroupedGrid
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelResult
open VeriTile.Triton FP.Structural KernelSource KernelExecution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

abbrev Lane := TileIndex [128, 128]
def flat (i : Lane) : Nat := i.1.val * 128 + i.2.1.val
def row (p : Parameters) (pid : Nat) (i : Lane) : Nat :=
  QuantTransposeFinite.tileRow (QuantTransposeFinite.gridM p.M)
    (QuantTransposeFinite.gridN p.N) pid * 128 + i.1.val
def col (p : Parameters) (pid : Nat) (i : Lane) : Nat :=
  QuantTransposeFinite.tileCol (QuantTransposeFinite.gridM p.M)
    (QuantTransposeFinite.gridN p.N) pid * 128 + i.2.1.val
def active (p : Parameters) (pid : Nat) (i : Lane) : Prop :=
  row p pid i < p.M ∧ col p pid i < p.N
instance (p : Parameters) (pid : Nat) (i : Lane) : Decidable (active p pid i) :=
  inferInstanceAs (Decidable (_ ∧ _))
def inputOffset (p : Parameters) (pid : Nat) (i : Lane) : Nat :=
  row p pid i * p.sam + col p pid i * p.san
def outputOffset (p : Parameters) (pid : Nat) (i : Lane) : Nat :=
  row p pid i + col p pid i * p.M

/-- Independent scalar formula in source order. Its arguments are kernel
inputs; no output or successful execution occurs in this definition. -/
def rounded {α : Type} (M : Algebra α) (round : α → BitVec 64)
    (value inverse : α) : BitVec 64 :=
  round (M.binary (some .fp32) .real .mul
    (M.literal (some .fp32) .real 127)
    (M.binary (some .fp32) .real .mul (M.fp32Load value) (M.fp32Load inverse)))

def quantized {α : Type} (M : Algebra α) (round : α → BitVec 64)
    (value inverse : α) : BitVec 8 :=
  BitVec.ofInt 8 (rounded M round value inverse).toInt

def word {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (s : State α) (i : Lane) : BitVec 64 :=
  rounded M round
    (if active p (s.pids 0) i then
      (s.mem ⟨p.x⟩ (inputOffset p (s.pids 0) i)).read .real
    else s.undef .real ⟨p.x⟩ (inputOffset p (s.pids 0) i))
    ((s.mem ⟨p.scale⟩ 0).read .real)

def written {α : Type} [Inhabited α] (M : Algebra α) (round : α → BitVec 64)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [128, 128]).foldl (fun t i =>
    if active p (s.pids 0) i then
      t.write ⟨p.output⟩ (outputOffset p (s.pids 0) i) (.mk .int (word M round p s i).toInt)
    else t) s

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (s : State α) :
    (execution M round p s).val.mem = (written M round p s).mem := by
  have hc : (127.0 : ℝ) = 127 := by norm_num
  have hm : p.M + 128 - 1 = p.M + 127 := by omega
  have hn : p.N + 128 - 1 = p.N + 127 := by omega
  simp [execution, written, word, rounded, row, col, active, inputOffset,
    outputOffset, QuantTransposeFinite.tileRow, QuantTransposeFinite.tileCol,
    QuantTransposeFinite.gridM, QuantTransposeFinite.gridN,
    QuantTransposeFinite.groupId, QuantTransposeFinite.groupSize,
    QuantTransposeFinite.groupWidth, hm, hn, hc, masked_scatter_memory,
    min_def_lt, Nat.mul_comm,
    Broadcast.leftIndex, Broadcast.rightIndex]

theorem flat_bound (i : Lane) : flat i < 128 * 128 := by
  have := i.1.isLt
  have := i.2.1.isLt
  dsimp [flat]
  omega

theorem flat_row (p : Parameters) (pid : Nat) (i : Lane) :
    QuantTransposeFinite.row p.M p.N pid (flat i) = row p pid i := by
  have hj := i.2.1.isLt
  have hd : flat i / 128 = i.1.val := by dsimp [flat]; omega
  simp only [QuantTransposeFinite.row, row, hd]

theorem flat_col (p : Parameters) (pid : Nat) (i : Lane) :
    QuantTransposeFinite.col p.M p.N pid (flat i) = col p pid i := by
  have hj := i.2.1.isLt
  have hd : flat i % 128 = i.2.1.val := by dsimp [flat]; omega
  simp only [QuantTransposeFinite.col, col, hd]

theorem flat_active (p : Parameters) (pid : Nat) (i : Lane) :
    QuantTransposeFinite.active p pid (flat i) ↔ active p pid i := by
  simp [QuantTransposeFinite.active, active, flat_row, flat_col]

theorem flat_output (p : Parameters) (pid : Nat) (i : Lane) :
    QuantTransposeFinite.offset p pid (flat i) = outputOffset p pid i := by
  simp [QuantTransposeFinite.offset, outputOffset, flat_row, flat_col, Nat.add_comm]

theorem flat_injective : Function.Injective flat := by
  rintro ⟨i, j, u⟩ ⟨k, l, v⟩ h
  have hi := i.isLt
  have hj := j.isLt
  have hk := k.isLt
  have hl := l.isLt
  have hik : i = k := Fin.ext (by dsimp [flat] at h; omega)
  have hjl : j = l := Fin.ext (by dsimp [flat] at h; omega)
  cases hik; cases hjl; rfl

theorem output_bound (p : Parameters) (pid : Nat) (i : Lane) (hi : active p pid i) :
    outputOffset p pid i < p.M * p.N := by
  simpa [flat_output, QuantTransposeFinite.Parameters.n] using
    (QuantTransposeFinite.active_index p pid (flat i) ((flat_active p pid i).2 hi)).1

theorem output_injective (p : Parameters) (pid : Nat) :
    ∀ i j : Lane, active p pid i → active p pid j →
      outputOffset p pid i = outputOffset p pid j → i = j := by
  intro i j hi hj he
  have er := QuantTransposeFinite.encode_decode p.M (row p pid i) (col p pid i) hi.1
  have er' := QuantTransposeFinite.encode_decode p.M (row p pid j) (col p pid j) hj.1
  have ee : col p pid i * p.M + row p pid i = col p pid j * p.M + row p pid j := by
    simpa [outputOffset, Nat.add_comm] using he
  have hr := congrArg (· % p.M) ee
  have hc := congrArg (· / p.M) ee
  dsimp only at hr hc
  rw [er.1, er'.1] at hr
  rw [er.2, er'.2] at hc
  apply flat_injective
  dsimp [row] at hr
  dsimp [col] at hc
  dsimp [flat]
  omega

theorem written_word {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (s : State α) (i : Lane)
    (hi : active p (s.pids 0) i) :
    (written M round p s).mem ⟨p.output⟩ (outputOffset p (s.pids 0) i) =
      .mk .int (word M round p s i).toInt :=
  masked_readback_active _ _ _ _ _ i hi (output_injective p (s.pids 0))

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α)
    (round : α → BitVec 64) (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (h : r ≠ ⟨p.output⟩ ∨ ∀ i : Lane, active p (s.pids 0) i → o ≠ outputOffset p (s.pids 0) i) :
    (written M round p s).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  exact h.imp id (fun h i _ hi => h i hi)

#axiomsClean execution_memory
#axiomsClean written_word
#axiomsClean written_frame
#specNonCircular quantized avoiding [KernelExecution.execution, KernelExecution.exec, KernelSource.kernel]
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelResult
