import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelExecution
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelResult
open VeriTile.Triton FP.Structural KernelSource KernelExecution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def inputOffset (p : Parameters) (program h c : Nat) : Nat :=
  program * (p.heads * p.columns) + h * p.columns + c
def codeOffset (p : Parameters) (dst h c : Nat) : Nat :=
  dst * (p.heads * p.columns) + p.columns * h + c
def scaleOffset (p : Parameters) (dst h : Nat) : Nat := dst * p.heads + h

def active (p : Parameters) (i : TileIndex [p.height,p.width]) : Prop :=
  i.1.val<p.heads ∧ i.2.1.val<p.columns
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : TileIndex [p.height, p.width]) : α :=
  M.fp32Load (if active p i then
    (s.mem "input" (inputOffset p (s.pids 0) i.1.val i.2.1.val)).read .real
    else M.literal (some .fp32) .real 0)

def maximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.height] :=
  M.reduceMax (some .fp32) ⟨1, by simp⟩ Bool.false
    (fun i => M.unary (some .fp32) (.abs .real) (loaded M p s i))

def scale {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (h : Fin p.height) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .div (maximum M p s (h, PUnit.unit))
      (M.literal (some .fp32) .real 127))

def denominator {α : Type} [Inhabited α] (M : Algebra α) (equal : α → α → Bool)
    (p : Parameters) (s : State α) (h : Fin p.height) : α :=
  if equal (maximum M p s (h, PUnit.unit)) (M.literal (some .fp32) .real 0) then
    M.literal (some .fp32) .real 1
  else M.cast (some .fp32) .real .real (scale M p s h)

def code {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α)
    (i : TileIndex [p.height, p.width]) : Int :=
  convert (M.binary (some .fp32) .real .div (loaded M p s i) (denominator M equal p s i.1))

def writtenCodes {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat) : State α :=
  (TileShape.allIndices [p.height, p.width]).foldl (fun t i =>
    if active p i then
      t.write "output" (codeOffset p dst i.1.val i.2.1.val) (.mk .int (code M convert equal p s i))
    else t) s

def written {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat) : State α :=
  (TileShape.allIndices [p.height, 1]).foldl (fun t i =>
    if i.1.val < p.heads then
      t.write "scale" (scaleOffset p dst i.1.val) (.mk .real (scale M p s i.1))
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
    inputOffset, codeOffset, scaleOffset, active, hd, hzero, hone, hlevels, masked_scatter_memory,
    index_toNat, index2_toNat,
    Broadcast.leftIndex, Broadcast.rightIndex]

theorem codeOffset_injective (p : Parameters) (dst : Nat)
    (i j : TileIndex [p.height,p.width]) (hi : active p i) (hj : active p j)
    (h : codeOffset p dst i.1.val i.2.1.val=codeOffset p dst j.1.val j.2.1.val) : i=j := by
  have he : p.columns*i.1.val+i.2.1.val=p.columns*j.1.val+j.2.1.val := by
    dsimp [codeOffset] at h
    omega
  have hc : i.2.1.val=j.2.1.val := by
    have hb := congrArg (· % p.columns) he
    simpa [Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt hi.2,Nat.mod_eq_of_lt hj.2] using hb
  have hh : i.1.val=j.1.val := by
    have hp : 0<p.columns := lt_of_le_of_lt (Nat.zero_le _) hi.2
    exact Nat.mul_left_cancel hp (by omega)
  exact Prod.ext (Fin.ext hh) (Prod.ext (Fin.ext hc) (Subsingleton.elim _ _))

theorem scaleOffset_injective (p : Parameters) (dst : Nat) :
    Function.Injective (fun i : TileIndex [p.height, 1] => scaleOffset p dst i.1.val) := by
  rintro ⟨i, c, u⟩ ⟨j, d, v⟩ h
  have hi : i = j := Fin.ext (Nat.add_left_cancel h)
  have hd : c = d := Subsingleton.elim _ _
  cases hi; cases hd; rfl

theorem written_code {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat)
    (i : TileIndex [p.height, p.width]) (hi : active p i) :
    (written M convert equal p s dst).mem "output" (codeOffset p dst i.1.val i.2.1.val) =
      .mk .int (code M convert equal p s i) := by
  unfold written
  rw [State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact masked_readback_active _ _ _ _ _ i hi (codeOffset_injective p dst)

theorem written_scale {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat)
    (i : TileIndex [p.height, 1]) (hi : i.1.val < p.heads) :
    (written M convert equal p s dst).mem "scale" (scaleOffset p dst i.1.val) =
      .mk .real (scale M p s i.1) :=
  State.masked_scatter_readback _ _ _ _ _ i hi (scaleOffset_injective p dst)

theorem written_frame {α : Type} [Inhabited α] (M : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (dst : Nat)
    (r : RegionName) (o : Nat)
    (codes : r ≠ "output" ∨ ∀ h c, h < p.heads → c < p.columns → o ≠ codeOffset p dst h c)
    (scales : r ≠ "scale" ∨ ∀ h, h < p.heads → o ≠ scaleOffset p dst h) :
    (written M convert equal p s dst).mem r o = s.mem r o := by
  unfold written
  rw [State.masked_scatter_frame]
  · unfold writtenCodes
    apply State.masked_scatter_frame
    rcases codes with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho _ _ hi.1 hi.2)
  · rcases scales with hr | ho
    · exact Or.inl hr
    · exact Or.inr (fun i _ hi => ho _ hi)

#axiomsClean execution_memory
#axiomsClean written_code
#axiomsClean written_scale
#axiomsClean written_frame
#specNonCircular code avoiding [execution, FP.Structural.exec, kernel]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelResult
