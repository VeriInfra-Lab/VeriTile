import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedView
import bench.tritonbench_g.relu_strided_buffer.Proofs.CappedGrid

/-!
Composition of actual per-tile execution contracts over the source capped grid.
The only execution premise below is a LOCAL tile theorem: it reads the current
input cell, produces one scalar result per active lane, and frames other cells.
The result for every output from the ORIGINAL input is proved by induction.
No full-output oracle, launch-result callback, or schedule-result premise occurs.

Both signs of a valid nonzero output stride are admitted. Input/output physical
footprints must be disjoint; equal region names are allowed when cells differ.
Native finite-width arithmetic and host failure behavior are separate obligations.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedGrid
open VeriTile.Triton SignedView CappedGrid

def TileOutside (output : View) (n start B : Nat) (r : RegionName) (a : Nat) : Prop :=
  r ≠ output.region ∨ ∀ lane : TileIndex [B], start + lane.1.val < n →
    a ≠ output.address (start + lane.1.val)

def Covered (B : Nat) (starts : List Nat) (i : Nat) : Prop :=
  ∃ start ∈ starts, ∃ lane : TileIndex [B], i = start + lane.1.val

def FootprintsDisjoint (input output : View) (n : Nat) : Prop :=
  ∀ i, i < n → ∀ j, j < n →
    input.region ≠ output.region ∨ input.address i ≠ output.address j

/-- An operation on ONE loaded scalar, later instantiated by the actual
typed `where(x > 0, x, 0)` and typed store. This is not an output function. -/
def TileContract (input output : View) (n B : Nat)
    (scalarCell : TileCarrier .fp32 → MemCell)
    (tileExec : Nat → BlockState → Option BlockState) : Prop :=
  ∀ start s, ∃ final, tileExec start s = some final ∧
    (∀ lane : TileIndex [B], start + lane.1.val < n →
      final.mem output.region (output.address (start + lane.1.val)) =
        scalarCell (s.readMemValue .fp32 input.region
          (input.address (start + lane.1.val)))) ∧
    (∀ r a, TileOutside output n start B r a → final.mem r a = s.mem r a)

def runTiles (tileExec : Nat → BlockState → Option BlockState) :
    List Nat → BlockState → Option BlockState
  | [], s => some s
  | start :: starts, s => (tileExec start s).bind (runTiles tileExec starts)

theorem input_outside_tile {input output : View} {n B start i : Nat}
    (hsep : FootprintsDisjoint input output n) (hi : i < n) :
    TileOutside output n start B input.region (input.address i) := by
  by_cases hr : input.region ≠ output.region
  · exact Or.inl hr
  · right
    intro lane hactive
    rcases hsep i hi (start + lane.1.val) hactive with hregion | hcell
    · exact False.elim (hr hregion)
    · exact hcell

/-- Success, all covered output values, preserved input cells, and exact frame
for an arbitrary finite list of tiles. This theorem derives the original-input
claim from the local frame and disjoint footprints. -/
theorem tiles_success {input output : View} {n B capacity : Nat}
    (hout : output.Valid n capacity) (hstride : output.stride ≠ 0)
    (hsep : FootprintsDisjoint input output n)
    (scalarCell : TileCarrier .fp32 → MemCell)
    (tileExec : Nat → BlockState → Option BlockState)
    (htile : TileContract input output n B scalarCell tileExec)
    (starts : List Nat) (s : BlockState) :
    ∃ final, runTiles tileExec starts s = some final ∧
      (∀ i, i < n → Covered B starts i →
        final.mem output.region (output.address i) =
          scalarCell (s.readMemValue .fp32 input.region (input.address i))) ∧
      (∀ i, i < n → final.mem input.region (input.address i) =
        s.mem input.region (input.address i)) ∧
      (∀ r a, (∀ start ∈ starts, TileOutside output n start B r a) →
        final.mem r a = s.mem r a) := by
  induction starts generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_, ?_⟩
    · intro i hi hcovered
      rcases hcovered with ⟨start, hmem, _⟩
      simp at hmem
    · intro i hi
      rfl
    · intro r a hframe
      rfl
  | cons start starts ih =>
    obtain ⟨middle, hexec, hvalue, hframe⟩ := htile start s
    obtain ⟨final, htail, htailValue, htailInput, htailFrame⟩ := ih middle
    have hinput (i : Nat) (hi : i < n) :
        middle.mem input.region (input.address i) = s.mem input.region (input.address i) :=
      hframe _ _ (input_outside_tile hsep hi)
    have hread (i : Nat) (hi : i < n) :
        middle.readMemValue .fp32 input.region (input.address i) =
          s.readMemValue .fp32 input.region (input.address i) := by
      simp only [BlockState.readMemValue, BlockState.readMemAs, hinput i hi]
    refine ⟨final, ?_, ?_, ?_, ?_⟩
    · simp only [runTiles, hexec, Option.bind_some, htail]
    · intro i hi hcovered
      by_cases hcoveredTail : Covered B starts i
      · rw [htailValue i hi hcoveredTail, hread i hi]
      · have hpreserve : final.mem output.region (output.address i) =
            middle.mem output.region (output.address i) := by
          apply htailFrame
          intro other hother
          right
          intro lane hactive heq
          have hindices := address_injective output hout hstride
            (a₁ := ⟨i, hi⟩) (a₂ := ⟨other + lane.1.val, hactive⟩) heq
          have hlogical : i = other + lane.1.val := congrArg Fin.val hindices
          exact hcoveredTail ⟨other, hother, lane, hlogical⟩
        rw [hpreserve]
        rcases hcovered with ⟨other, hother, lane, hlogical⟩
        rcases List.mem_cons.mp hother with heq | hmem
        · subst other
          subst i
          exact hvalue lane hi
        · exact False.elim (hcoveredTail ⟨other, hmem, lane, hlogical⟩)
    · intro i hi
      rw [htailInput i hi, hinput i hi]
    · intro r a houtside
      rw [htailFrame r a (fun other hmem => houtside other (List.mem_cons_of_mem start hmem))]
      exact hframe r a (houtside start (List.mem_cons_self))

def cappedStarts (n B : Nat) : List Nat :=
  (List.range (numCTAs n B)).flatMap fun pid =>
    (List.range (tilesPerCTA n B)).map fun step =>
      (pid + step * numCTAs n B) * B

theorem capped_covers {n B i : Nat} (hB : 0 < B) (hi : i < n) :
    Covered B (cappedStarts n B) i := by
  have h := logical_index_covered hB hi
  dsimp at h
  refine ⟨(i / B % numCTAs n B + i / B / numCTAs n B * numCTAs n B) * B,
    ?_, ⟨⟨i % B, h.2.2.1⟩, PUnit.unit⟩, ?_⟩
  · apply List.mem_flatMap.mpr
    refine ⟨i / B % numCTAs n B, List.mem_range.mpr h.1, ?_⟩
    apply List.mem_map.mpr
    exact ⟨i / B / numCTAs n B, List.mem_range.mpr h.2.1, rfl⟩
  · simpa only [logicalIndex] using h.2.2.2.symm

theorem covered_of_perm {B i : Nat} {first second : List Nat}
    (hperm : first.Perm second) (hcovered : Covered B second i) :
    Covered B first i := by
  rcases hcovered with ⟨start, hmem, lane, heq⟩
  exact ⟨start, hperm.mem_iff.mpr hmem, lane, heq⟩

/-- Output address equality for active coordinates forces the SAME CTA,
iteration and lane. Valid negative strides are covered by the same theorem. -/
theorem active_output_unique (output : View) {n capacity B c p j l p' j' l' : Nat}
    (hout : output.Valid n capacity) (hstride : output.stride ≠ 0)
    (hB : 0 < B) (hc : 0 < c)
    (hp : p < c) (hp' : p' < c) (hl : l < B) (hl' : l' < B)
    (hi : logicalIndex B c p j l < n) (hi' : logicalIndex B c p' j' l' < n)
    (heq : output.address (logicalIndex B c p j l) =
      output.address (logicalIndex B c p' j' l')) :
    p = p' ∧ j = j' ∧ l = l' := by
  have hindices := address_injective output hout hstride
    (a₁ := ⟨logicalIndex B c p j l, hi⟩)
    (a₂ := ⟨logicalIndex B c p' j' l', hi'⟩) heq
  exact logical_index_injective hB hc hp hp' hl hl' (congrArg Fin.val hindices)

/-- Any complete ordering of the source's capped tiles produces all logical
outputs from the ORIGINAL input cells, preserves every input cell, and frames
every cell outside the complete output footprint. The local execution premise
is intended to be discharged by `SignedTile.tile_success`. -/
theorem capped_grid_success {input output : View} {n B inputCapacity outputCapacity : Nat}
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride ≠ 0) (hB : 0 < B)
    (hsep : FootprintsDisjoint input output n)
    (scalarCell : TileCarrier .fp32 → MemCell)
    (tileExec : Nat → BlockState → Option BlockState)
    (htile : TileContract input output n B scalarCell tileExec)
    (starts : List Nat) (hschedule : starts.Perm (cappedStarts n B)) (s : BlockState) :
    ∃ final, runTiles tileExec starts s = some final ∧
      (∀ i, i < n → final.mem output.region (output.address i) =
        scalarCell (s.readMemValue .fp32 input.region (input.address i))) ∧
      (∀ i, i < n → final.mem input.region (input.address i) =
        s.mem input.region (input.address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ output.address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → input.address i < inputCapacity ∧ output.address i < outputCapacity) := by
  obtain ⟨final, hexec, hvalue, hinput, hframe⟩ :=
    tiles_success hout hstride hsep scalarCell tileExec htile starts s
  refine ⟨final, hexec, ?_, hinput, ?_, ?_⟩
  · intro i hi
    exact hvalue i hi (covered_of_perm hschedule (capped_covers hB hi))
  · intro r a houtside
    apply hframe r a
    intro start hstart
    rcases houtside with hr | hc
    · exact Or.inl hr
    · exact Or.inr (fun lane hactive => hc _ hactive)
  · intro i hi
    exact ⟨address_lt input hin hi, address_lt output hout hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.SignedGrid
