import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-!
Signed rank-one view addressing for `relu_strided_buffer.py`.

The original `BlockPtr.strides : List Nat` cannot encode negative strides.
This local lowering uses the existing signed scalar arithmetic and direct
memory-access operations. It accumulates the COMPLETE signed cell offset
before `Int.toNat`, and retains the source's logical boundary mask.

This file proves address, load and store facts, not a native compiler
refinement or a complete wrapper theorem. FP32 is the sole storage channel
used below. Its mathematical carrier is not a bit-accurate IEEE model.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedView
open VeriTile.Triton

structure View where
  region : RegionName
  base : Int
  stride : Int

def View.signedAddress (v : View) (i : Nat) : Int :=
  v.base + (i : Int) * v.stride

def View.address (v : View) (i : Nat) : Nat := (v.signedAddress i).toNat

/-- Ordinary view bounds; contains no desired output or execution result. -/
def View.Valid (v : View) (n capacity : Nat) : Prop :=
  ∀ i, i < n → 0 ≤ v.signedAddress i ∧ v.signedAddress i < (capacity : Int)

theorem address_cast (v : View) {i : Nat} (hi : 0 ≤ v.signedAddress i) :
    (v.address i : Int) = v.signedAddress i := Int.toNat_of_nonneg hi

theorem address_lt (v : View) {n capacity i : Nat}
    (hv : v.Valid n capacity) (hi : i < n) : v.address i < capacity := by
  have h := (hv i hi).2
  rw [← address_cast v (hv i hi).1] at h
  exact_mod_cast h

/-- A nonzero output stride is injective for both signs, provided active
addresses are valid. Negative inactive addresses are never used for stores. -/
theorem address_injective (v : View) {n capacity : Nat}
    (hv : v.Valid n capacity) (hs : v.stride ≠ 0) :
    Function.Injective (fun i : Fin n => v.address i.val) := by
  intro i j hij
  have heq := congrArg (fun a : Nat => (a : Int)) hij
  change (v.address i.val : Int) = (v.address j.val : Int) at heq
  rw [address_cast v (hv i.val i.isLt).1,
      address_cast v (hv j.val j.isLt).1] at heq
  have hm : (i.val : Int) * v.stride = (j.val : Int) * v.stride := by
    simpa only [View.signedAddress, add_right_inj] using heq
  have hi : (i.val : Int) = (j.val : Int) := mul_right_cancel₀ hs hm
  apply Fin.ext
  exact_mod_cast hi

theorem nat_address (r : RegionName) (base stride i : Nat) :
    (View.mk r base stride).address i = base + i * stride := by
  simp only [View.address, View.signedAddress, ← Int.natCast_mul,
    ← Int.natCast_add, Int.toNat_natCast]

/-- Exact agreement with the old natural-stride block-pointer address. -/
theorem nat_block_address (r : RegionName)
    (base stride n B start lane : Nat) :
    (View.mk r base stride).address (start + lane) =
      BlockPtr.address
        { region := r, baseOffset := base, parentShape := [n],
          blockShape := [B], strides := [stride], offsets := [start] } [lane] := by
  rw [nat_address, BlockPtr.address_1d]

/-- Reverse views use a positive base, not per-addend Nat truncation. -/
theorem reverse_address (r : RegionName) {n i : Nat} (hi : i < n) :
    (View.mk r ((n : Int) - 1) (-1)).address i = n - 1 - i := by
  simp only [View.address, View.signedAddress]
  omega

def indexOp (start B : Nat) : Op .nat [B] :=
  .add .nat .scalarL (.constNat start) (.arange B)

def addressOp (v : View) (start B : Nat) : Op .nat [B] :=
  .castIntToNat (.add .int .scalarL (.constInt v.base)
    (.mul .int .scalarR (.castNatToInt (indexOp start B)) (.constInt v.stride)))

def maskOp (n start B : Nat) : Op .bool [B] :=
  .lt .nat .scalarR (indexOp start B) (.constNat n)

theorem indexOp_eval (start B : Nat) (s : BlockState) :
    evalOp (indexOp start B) s =
      some (⟨fun i : TileIndex [B] => start + i.1.val⟩ : Tile .nat [B]) := by
  simp only [indexOp, evalOp, bind, Option.bind]
  rfl

theorem addressOp_eval (v : View) (start B : Nat) (s : BlockState) :
    evalOp (addressOp v start B) s =
      some (⟨fun i : TileIndex [B] => v.address (start + i.1.val)⟩ : Tile .nat [B]) := by
  simp only [addressOp, evalOp, indexOp_eval, bind, Option.bind]
  rfl

theorem maskOp_eval (n start B : Nat) (s : BlockState) :
    evalOp (maskOp n start B) s =
      some (⟨fun i : TileIndex [B] => decide (start + i.1.val < n)⟩ : Tile .bool [B]) := by
  simp only [maskOp, evalOp, indexOp_eval, bind, Option.bind]
  rfl

/-- FP32 direct-access lowering of the source boundary-checked block load.
Omitted padding is kept unspecified. -/
def load32 (v : View) (n start B : Nat) : Op .fp32 [B] :=
  .load .fp32 (.region v.region (addressOp v start B))
    (.mask (maskOp n start B)) .unspecified

theorem load32_eval (v : View) (n start B : Nat) (s : BlockState) :
    evalOp (load32 v n start B) s = some
      (⟨fun i : TileIndex [B] =>
        if start + i.1.val < n then
          s.readMemValue .fp32 v.region (v.address (start + i.1.val))
        else s.paddingValue .fp32 .unspecified v.region
          (v.address (start + i.1.val))⟩ : Tile .fp32 [B]) := by
  simp only [load32, evalOp, addressOp_eval, maskOp_eval, bind, Option.bind]
  refine congrArg some ?_
  ext i
  by_cases hi : start + i.1.val < n <;>
    simp [hi, BlockState.paddingValue, BlockState.unspecifiedValue]

/-- The source store consumes its already computed FP32 register; no callback
supplies a kernel output. -/
def store32 (v : View) (n start B : Nat) (value : RegName) : Stmt :=
  .store .fp32 [B] (.region v.region (addressOp v start B))
    (.ref .fp32 [B] value) (.mask (maskOp n start B))

def scatter32 (v : View) (n start B : Nat)
    (values : Tile .fp32 [B]) (s : BlockState) : BlockState :=
  (TileShape.allIndices [B]).foldl
    (fun acc i => if start + i.1.val < n then
      acc.writeMemTyped .fp32 v.region (v.address (start + i.1.val)) (values.data i)
      else acc) s

theorem store32_step (v : View) (n start B : Nat) (name : RegName)
    (values : Tile .fp32 [B]) (s : BlockState)
    (hvalues : s.regs .fp32 [B] name = some values) :
    stepStmt (store32 v n start B name) s =
      some (scatter32 v n start B values s) := by
  simp only [store32, stepStmt, evalOp_ref, hvalues, addressOp_eval,
    maskOp_eval, bind, Option.bind, Option.map, scatter32]
  refine congrArg some ?_
  apply List.foldl_ext
  intro acc i _
  by_cases hi : start + i.1.val < n <;> simp [hi]

/-- Exact typed-cell frame outside the active output footprint. This also
covers padding lanes whose signed address would be negative. -/
theorem scatter32_frame (v : View) (n start B : Nat)
    (values : Tile .fp32 [B]) (s : BlockState) (r : RegionName) (a : Nat)
    (houtside : r ≠ v.region ∨ ∀ i : TileIndex [B],
      start + i.1.val < n → a ≠ v.address (start + i.1.val)) :
    (scatter32 v n start B values s).mem r a = s.mem r a := by
  unfold scatter32
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i lanes ih =>
    simp only [List.foldl_cons]
    rw [ih]
    by_cases hi : start + i.1.val < n
    · simp only [hi, if_true, BlockState.writeMemTyped, BlockState.writeMemAs]
      rw [if_neg]
      rintro ⟨hr, ha⟩
      rcases houtside with hr' | ha'
      · exact hr' hr
      · exact ha' i hi ha
    · simp only [hi, if_false]

def storedCell32 (value : TileCarrier .fp32) : MemCell :=
  MemCell.of .fp32 (FloatDType.fp32.ofReal (FloatDType.fp32.storeValue value))

private theorem fold32_preserves_written (v : View) (n start B : Nat)
    (values : Tile .fp32 [B]) (target : TileIndex [B])
    (hnocollision : ∀ i : TileIndex [B], start + i.1.val < n →
      v.address (start + i.1.val) = v.address (start + target.1.val) → i = target)
    (lanes : List (TileIndex [B])) (s : BlockState)
    (hwritten : s.mem v.region (v.address (start + target.1.val)) =
      storedCell32 (values.data target)) :
    (lanes.foldl
      (fun acc i => if start + i.1.val < n then
        acc.writeMemTyped .fp32 v.region (v.address (start + i.1.val)) (values.data i)
        else acc) s).mem v.region (v.address (start + target.1.val)) =
      storedCell32 (values.data target) := by
  induction lanes generalizing s with
  | nil => exact hwritten
  | cons i lanes ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hi : start + i.1.val < n
    · simp only [hi, if_true, BlockState.writeMemTyped, BlockState.writeMemAs]
      by_cases ha : v.address (start + i.1.val) = v.address (start + target.1.val)
      · have hit := hnocollision i hi ha
        subst i
        simp [storedCell32]
      · simp [Ne.symm ha, hwritten]
    · simpa only [hi, if_false] using hwritten

/-- Every active lane's concrete typed store is observable at its signed
physical address. Nonzero negative output strides are admitted. -/
theorem scatter32_readback (v : View) {n capacity start B : Nat}
    (hv : v.Valid n capacity) (hs : v.stride ≠ 0)
    (values : Tile .fp32 [B]) (s : BlockState) (target : TileIndex [B])
    (htarget : start + target.1.val < n) :
    (scatter32 v n start B values s).mem v.region
      (v.address (start + target.1.val)) = storedCell32 (values.data target) := by
  have hnocollision : ∀ i : TileIndex [B], start + i.1.val < n →
      v.address (start + i.1.val) = v.address (start + target.1.val) → i = target := by
    intro i hi ha
    have heq := address_injective v hv hs
      (a₁ := ⟨start + i.1.val, hi⟩) (a₂ := ⟨start + target.1.val, htarget⟩) ha
    have hval := congrArg Fin.val heq
    apply Prod.ext
    · apply Fin.ext
      exact Nat.add_left_cancel hval
    · exact Subsingleton.elim _ _
  obtain ⟨before, after, hall⟩ :=
    List.append_of_mem (TileShape.mem_allIndices [B] target)
  unfold scatter32
  rw [hall, List.foldl_append, List.foldl_cons]
  simp only [htarget, if_true]
  apply fold32_preserves_written v n start B values target hnocollision
  simp [BlockState.writeMemTyped, BlockState.writeMemAs, storedCell32]

theorem store32_success_readback (v : View) {n capacity start B : Nat}
    (hv : v.Valid n capacity) (hs : v.stride ≠ 0)
    (name : RegName) (values : Tile .fp32 [B]) (s : BlockState)
    (hvalues : s.regs .fp32 [B] name = some values) :
    ∃ final, stepStmt (store32 v n start B name) s = some final ∧
      (∀ i : TileIndex [B], start + i.1.val < n →
        final.mem v.region (v.address (start + i.1.val)) = storedCell32 (values.data i)) ∧
      (∀ r a, (r ≠ v.region ∨ ∀ i : TileIndex [B],
        start + i.1.val < n → a ≠ v.address (start + i.1.val)) →
        final.mem r a = s.mem r a) := by
  refine ⟨scatter32 v n start B values s, store32_step v n start B name values s hvalues,
    ?_, ?_⟩
  · intro i hi
    exact scatter32_readback v hv hs values s i hi
  · intro r a ha
    exact scatter32_frame v n start B values s r a ha

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedView
