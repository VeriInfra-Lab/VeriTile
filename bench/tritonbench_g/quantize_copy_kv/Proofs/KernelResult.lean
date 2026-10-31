import bench.tritonbench_g.quantize_copy_kv.Proofs.KernelExecution
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelResult
open VeriTile.Triton FP.Structural KernelSource KernelExecution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def inputOffset (p : Parameters) (program h c : Nat) : Nat :=
  program * (p.heads * p.width) + h * p.width + c
def codeOffset (p : Parameters) (dst h c : Nat) : Nat :=
  dst * (p.heads * p.width) + p.width * h + c
def scaleOffset (p : Parameters) (dst h : Nat) : Nat := dst * p.heads + h

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.height, p.width]) : α :=
  M.fp32Load (if i.1.val < p.heads then
    (s.mem "input" (inputOffset p (s.pids 0) i.1.val i.2.1.val)).read .real
    else M.literal (some .fp32) .real 0)

def maximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.height] :=
  M.reduceMax (some .fp32) ⟨1, by simp⟩ Bool.false
    (fun i => M.unary (some .fp32) (.abs .real) (loaded M p s i))

def scale {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (h : Fin p.height) : α :=
  M.cast (some .fp32) .real .fp16
    (M.binary (some .fp32) .real .div (maximum M p s (h, PUnit.unit))
      (M.literal (some .fp32) .real 127))

def denominator {α : Type} [Inhabited α] (M : Algebra α) (equal : α → α → Bool)
    (p : Parameters) (s : State α) (h : Fin p.height) : α :=
  if equal (maximum M p s (h, PUnit.unit)) (M.literal (some .fp32) .real 0) then
    M.literal (some .fp32) .real 1
  else M.cast (some .fp32) .fp16 .real (scale M p s h)

def code {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α)
    (i : TileIndex [p.height, p.width]) : Int :=
  convert (M.binary (some .fp32) .real .div (loaded M p s i) (denominator M equal p s i.1))

def writtenCodes {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat) : State α :=
  (TileShape.allIndices [p.height, p.width]).foldl (fun t i =>
    if i.1.val < p.heads then
      t.write "output" (codeOffset p dst i.1.val i.2.1.val) (.mk .int (code M convert equal p s i))
    else t) s

def written {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat) : State α :=
  (TileShape.allIndices [p.height, 1]).foldl (fun t i =>
    if i.1.val < p.heads then
      t.write "scale" (scaleOffset p dst i.1.val) (.mk .fp16 (scale M p s i.1))
    else t) (writtenCodes M convert equal p s dst)

/-- The actual successful kernel has exactly the independent initial-memory
quantization outputs. The input-only metadata equation supplies its destination. -/
theorem index_toNat (a b c d e f : Nat) :
    Int.toNat ((a : Int) * (b * c) + d * e + f) = a * (b * c) + d * e + f := by
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

theorem index2_toNat (a b c : Nat) :
    Int.toNat ((a : Int) * b + c) = a * b + c := by
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (hw : 0 < p.width)
    (dst : Nat) (hd : (s.mem "destination" (s.pids 0)).read .int = (dst : Int)) :
    (execution M convert equal p s hw).val.mem = (written M convert equal p s dst).mem := by
  have hzero : (0.0 : ℝ) = 0 := by norm_num
  have hone : (1.0 : ℝ) = 1 := by norm_num
  have hlevels : (127.0 : ℝ) = 127 := by norm_num
  simp [execution, written, writtenCodes, loaded, maximum, scale, denominator, code,
    inputOffset, codeOffset, scaleOffset, hd, hzero, hone, hlevels, masked_scatter_memory,
    index_toNat, index2_toNat,
    Broadcast.leftIndex, Broadcast.rightIndex]

theorem codeOffset_injective (p : Parameters) (dst : Nat) :
    Function.Injective (fun i : TileIndex [p.height, p.width] =>
      codeOffset p dst i.1.val i.2.1.val) := by
  intro i j h
  change dst * (p.heads * p.width) + p.width * i.1.val + i.2.1.val =
    dst * (p.heads * p.width) + p.width * j.1.val + j.2.1.val at h
  have he : p.width * i.1.val + i.2.1.val = p.width * j.1.val + j.2.1.val := by omega
  have hc : i.2.1.val = j.2.1.val := by
    have hb := congrArg (· % p.width) he
    simpa [codeOffset, Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt i.2.1.isLt,
      Nat.mod_eq_of_lt j.2.1.isLt] using hb
  have hh : i.1.val = j.1.val := by
    have hp : 0 < p.width := lt_of_le_of_lt (Nat.zero_le _) i.2.1.isLt
    exact Nat.mul_left_cancel hp (by omega)
  rcases i with ⟨i, c, u⟩
  rcases j with ⟨j, d, v⟩
  have hi : i = j := Fin.ext hh
  have hd : c = d := Fin.ext hc
  cases hi; cases hd; rfl

theorem scaleOffset_injective (p : Parameters) (dst : Nat) :
    Function.Injective (fun i : TileIndex [p.height, 1] => scaleOffset p dst i.1.val) := by
  rintro ⟨i, c, u⟩ ⟨j, d, v⟩ h
  have hi : i = j := Fin.ext (Nat.add_left_cancel h)
  have hd : c = d := Subsingleton.elim _ _
  cases hi; cases hd; rfl

theorem written_code {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat)
    (i : TileIndex [p.height, p.width]) (hi : i.1.val < p.heads) :
    (written M convert equal p s dst).mem "output" (codeOffset p dst i.1.val i.2.1.val) =
      .mk .int (code M convert equal p s i) := by
  unfold written
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact State.masked_scatter_readback _ _ _ _ _ i hi (codeOffset_injective p dst)

theorem written_scale {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat)
    (i : TileIndex [p.height, 1]) (hi : i.1.val < p.heads) :
    (written M convert equal p s dst).mem "scale" (scaleOffset p dst i.1.val) =
      .mk .fp16 (scale M p s i.1) :=
  State.masked_scatter_readback _ _ _ _ _ i hi (scaleOffset_injective p dst)

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat)
    (r : RegionName) (o : Nat)
    (codes : r ≠ "output" ∨ ∀ h c, h < p.heads → c < p.width → o ≠ codeOffset p dst h c)
    (scales : r ≠ "scale" ∨ ∀ h, h < p.heads → o ≠ scaleOffset p dst h) :
    (written M convert equal p s dst).mem r o = s.mem r o := by
  unfold written
  rw [State.masked_scatter_frame]
  · unfold writtenCodes
    apply State.masked_scatter_frame
    rcases codes with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho _ _ hi i.2.1.isLt)
  · rcases scales with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho _ hi)

#axiomsClean execution_memory
#axiomsClean written_code
#axiomsClean written_scale
#axiomsClean written_frame
#specNonCircular code avoiding [FP.Structural.exec,kernel]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelResult
