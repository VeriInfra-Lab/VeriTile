import bench.tritonbench_g.relu_strided_buffer.Proofs.SignedGrid
import VeriTile.Triton.Float.Structural

/-!
Opaque floating values and exact cell frames over the original capped grid.
The execution premise is only a local per-tile theorem. Composition derives
all results from the launch's original input cells, without numerical laws or
a supplied complete output. Tile IDs are the actual pid + j * num_ctas values.
-/
namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralGrid
open VeriTile.Triton SignedView CappedGrid FP.Structural

def Covered (B : Nat) (tiles : List Nat) (i : Nat) : Prop :=
  ∃ t ∈ tiles, ∃ lane : TileIndex [B], i = t * B + lane.1.val

def TileContract {α : Type} [Inhabited α] (input output : View) (n B : Nat)
    (scalarCell : α → Cell α) (tileExec : Nat → State α → Option (State α)) : Prop :=
  ∀ t s, ∃ final, tileExec t s = some final ∧
    (∀ lane : TileIndex [B], t * B + lane.1.val < n →
      final.mem output.region (output.address (t * B + lane.1.val)) =
        scalarCell ((s.mem input.region (input.address (t * B + lane.1.val))).read .fp32)) ∧
    (∀ r a, SignedGrid.TileOutside output n (t * B) B r a → final.mem r a = s.mem r a)

def runTiles {α : Type} (tileExec : Nat → State α → Option (State α)) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | t :: tiles, s => (tileExec t s).bind (runTiles tileExec tiles)

theorem runTiles_append {α : Type} (tileExec : Nat → State α → Option (State α))
    (first second : List Nat) (s : State α) :
    runTiles tileExec (first ++ second) s =
      (runTiles tileExec first s).bind (runTiles tileExec second) := by
  induction first generalizing s with
  | nil => rfl
  | cons t rest ih =>
    simp only [List.cons_append, runTiles]
    cases h : tileExec t s with
    | none => rfl
    | some middle => simpa using ih middle

theorem tiles_success {α : Type} [Inhabited α] {input output : View} {n B capacity : Nat}
    (hout : output.Valid n capacity) (hstride : output.stride ≠ 0)
    (hsep : SignedGrid.FootprintsDisjoint input output n)
    (scalarCell : α → Cell α) (tileExec : Nat → State α → Option (State α))
    (htile : TileContract input output n B scalarCell tileExec)
    (tiles : List Nat) (s : State α) :
    ∃ final, runTiles tileExec tiles s = some final ∧
      (∀ i, i < n → Covered B tiles i →
        final.mem output.region (output.address i) =
          scalarCell ((s.mem input.region (input.address i)).read .fp32)) ∧
      (∀ i, i < n → final.mem input.region (input.address i) =
        s.mem input.region (input.address i)) ∧
      (∀ r a, (∀ t ∈ tiles, SignedGrid.TileOutside output n (t * B) B r a) →
        final.mem r a = s.mem r a) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_, ?_⟩
    · intro i hi hcovered
      rcases hcovered with ⟨t, hmem, _⟩
      simp at hmem
    · intro i hi
      rfl
    · intro r a hframe
      rfl
  | cons t tiles ih =>
    obtain ⟨middle, hexec, hvalue, hframe⟩ := htile t s
    obtain ⟨final, htail, htailValue, htailInput, htailFrame⟩ := ih middle
    have hinput (i : Nat) (hi : i < n) :
        middle.mem input.region (input.address i) = s.mem input.region (input.address i) :=
      hframe _ _ (SignedGrid.input_outside_tile hsep hi)
    refine ⟨final, ?_, ?_, ?_, ?_⟩
    · simp only [runTiles, hexec, Option.bind_some, htail]
    · intro i hi hcovered
      by_cases hcoveredTail : Covered B tiles i
      · rw [htailValue i hi hcoveredTail, hinput i hi]
      · have hpreserve : final.mem output.region (output.address i) =
            middle.mem output.region (output.address i) := by
          apply htailFrame
          intro other hother
          right
          intro lane hactive heq
          have hindices := address_injective output hout hstride
            (a₁ := ⟨i, hi⟩) (a₂ := ⟨other * B + lane.1.val, hactive⟩) heq
          exact hcoveredTail ⟨other, hother, lane, congrArg Fin.val hindices⟩
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
      rw [htailFrame r a (fun other hmem => houtside other (List.mem_cons_of_mem t hmem))]
      exact hframe r a (houtside t List.mem_cons_self)

def cappedTileIds (n B : Nat) : List Nat :=
  (List.range (numCTAs n B)).flatMap fun pid =>
    (List.range (tilesPerCTA n B)).map fun j => pid + j * numCTAs n B

theorem capped_covers {n B i : Nat} (hB : 0 < B) (hi : i < n) :
    Covered B (cappedTileIds n B) i := by
  have h := logical_index_covered hB hi
  dsimp at h
  refine ⟨i / B % numCTAs n B + i / B / numCTAs n B * numCTAs n B,
    ?_, ⟨⟨i % B, h.2.2.1⟩, PUnit.unit⟩, ?_⟩
  · apply List.mem_flatMap.mpr
    refine ⟨i / B % numCTAs n B, List.mem_range.mpr h.1, ?_⟩
    apply List.mem_map.mpr
    exact ⟨i / B / numCTAs n B, List.mem_range.mpr h.2.1, rfl⟩
  · simpa only [logicalIndex] using h.2.2.2.symm

theorem capped_grid_success {α : Type} [Inhabited α] {input output : View}
    {n B inputCapacity outputCapacity : Nat}
    (hin : input.Valid n inputCapacity) (hout : output.Valid n outputCapacity)
    (hstride : output.stride ≠ 0) (hB : 0 < B)
    (hsep : SignedGrid.FootprintsDisjoint input output n)
    (scalarCell : α → Cell α) (tileExec : Nat → State α → Option (State α))
    (htile : TileContract input output n B scalarCell tileExec)
    (tiles : List Nat) (hschedule : tiles.Perm (cappedTileIds n B)) (s : State α) :
    ∃ final, runTiles tileExec tiles s = some final ∧
      (∀ i, i < n → final.mem output.region (output.address i) =
        scalarCell ((s.mem input.region (input.address i)).read .fp32)) ∧
      (∀ i, i < n → final.mem input.region (input.address i) = s.mem input.region (input.address i)) ∧
      (∀ r a, (r ≠ output.region ∨ ∀ i, i < n → a ≠ output.address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < n → input.address i < inputCapacity ∧ output.address i < outputCapacity) := by
  obtain ⟨final, hexec, hvalue, hinput, hframe⟩ :=
    tiles_success hout hstride hsep scalarCell tileExec htile tiles s
  refine ⟨final, hexec, ?_, hinput, ?_, ?_⟩
  · intro i hi
    apply hvalue i hi
    obtain ⟨t, hmem, lane, heq⟩ := capped_covers hB hi
    exact ⟨t, hschedule.mem_iff.mpr hmem, lane, heq⟩
  · intro r a houtside
    apply hframe r a
    intro t hmem
    rcases houtside with hr | hc
    · exact Or.inl hr
    · exact Or.inr (fun lane hactive => hc _ hactive)
  · intro i hi
    exact ⟨address_lt input hin hi, address_lt output hout hi⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.StructuralGrid
