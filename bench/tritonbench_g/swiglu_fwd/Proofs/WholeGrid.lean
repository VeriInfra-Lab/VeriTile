import bench.tritonbench_g.swiglu_fwd.Proofs.WholeTile

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Ownership of the writable logical cells; X and Y need not be distinct. -/
structure Ownership (c : Layout) : Prop where
  xSeparate : c.X ≠ c.OUT
  ySeparate : c.Y ≠ c.OUT
  outputUnique : ∀ r q r' q', r < c.rows → q < c.cols → r' < c.rows → q' < c.cols →
    r * c.so + q = r' * c.so + q' → r = r' ∧ q = q'

def tileExec {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (B : Nat)
    (t : Nat × Nat) (s : State α) : Option (State α) :=
  run M (c.source B).surfaceBody (tileState M c B t.1 t.2 s)

def runTiles {α : Type} [Inhabited α] (M : Algebra α) (c : Layout) (B : Nat) :
    List (Nat × Nat) → State α → Option (State α)
  | [], s => some s
  | t :: ts, s => (tileExec M c B t s).bind (runTiles M c B ts)

def Covered (B : Nat) (tiles : List (Nat × Nat)) (r q : Nat) : Prop :=
  ∃ b, (r,b) ∈ tiles ∧ ∃ lane : TileIndex [B], q = b * B + lane.1.val

theorem tiles_result {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (own : Ownership c) (tiles : List (Nat × Nat))
    (valid : ∀ t ∈ tiles, t.1 < c.rows) (s : State α) :
    ∃ final, runTiles M c B tiles s = some final ∧
      (∀ r q, r < c.rows → q < c.cols → Covered B tiles r q →
        final.mem c.OUT (r*c.so+q) = Cell.mk .real (scalar M
          ((s.mem c.X (r*c.sx+q)).read .real) ((s.mem c.Y (r*c.sy+q)).read .real))) ∧
      (∀ region a, (∀ t ∈ tiles, Outside c B t.1 t.2 region a) →
        final.mem region a = s.mem region a) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s, rfl, ?_, ?_⟩
    · intro r q hr hq hcovered
      rcases hcovered with ⟨b,hb,_⟩
      simp at hb
    · intro region a _
      rfl
  | cons t tiles ih =>
    let middle := (sourceRun M c B t.1 t.2 s).val
    have runHead : tileExec M c B t s = some middle := (sourceRun M c B t.1 t.2 s).property
    obtain ⟨final, runTail, valueTail, frameTail⟩ :=
      ih (fun other hmem => valid other (List.mem_cons_of_mem t hmem)) middle
    have xframe (r q : Nat) : middle.mem c.X (r*c.sx+q) = s.mem c.X (r*c.sx+q) :=
      tile_frame M c B t.1 t.2 s _ _ (Or.inl own.xSeparate)
    have yframe (r q : Nat) : middle.mem c.Y (r*c.sy+q) = s.mem c.Y (r*c.sy+q) :=
      tile_frame M c B t.1 t.2 s _ _ (Or.inl own.ySeparate)
    refine ⟨final, ?_, ?_, ?_⟩
    · simp only [runTiles, runHead, Option.bind_some, runTail]
    · intro r q hr hq covered
      by_cases tail : Covered B tiles r q
      · rw [valueTail r q hr hq tail, xframe, yframe]
      · have preserve : final.mem c.OUT (r*c.so+q) = middle.mem c.OUT (r*c.so+q) := by
          apply frameTail
          intro other hother
          right
          intro lane active equal
          have same := own.outputUnique r q other.1 (other.2*B+lane.1.val)
            hr hq (valid other (List.mem_cons_of_mem t hother)) active equal
          apply tail
          refine ⟨other.2, ?_, lane, same.2⟩
          simpa only [same.1] using hother
        rw [preserve]
        rcases covered with ⟨b,hb,lane,heq⟩
        rcases List.mem_cons.mp hb with head | rest
        · cases head
          subst q
          exact tile_readback M c B r b s lane hq
        · exact False.elim (tail ⟨b,rest,lane,heq⟩)
    · intro region a outside
      rw [frameTail region a (fun other hmem => outside other (List.mem_cons_of_mem t hmem))]
      exact tile_frame M c B t.1 t.2 s region a (outside t List.mem_cons_self)

def grid (c : Layout) (B : Nat) : List (Nat × Nat) :=
  (List.range c.rows).flatMap fun r =>
    (List.range ((c.cols+B-1)/B)).map fun b => (r,b)

theorem grid_valid (c : Layout) (B : Nat) : ∀ t ∈ grid c B, t.1 < c.rows := by
  intro t ht
  rcases List.mem_flatMap.mp ht with ⟨r,hr,hb⟩
  rcases List.mem_map.mp hb with ⟨b,_,heq⟩
  subst t
  exact List.mem_range.mp hr

theorem grid_covers (c : Layout) (B : Nat) (positive : 0 < B)
    (r q : Nat) (hr : r < c.rows) (hq : q < c.cols) : Covered B (grid c B) r q := by
  have hm := Nat.mod_lt q positive
  have hcover : q / B < (c.cols+B-1)/B := by
    have hqdiv := Nat.div_add_mod q B
    have hndiv := Nat.div_add_mod (c.cols+B-1) B
    have hnmod := Nat.mod_lt (c.cols+B-1) positive
    by_contra h
    have le : (c.cols+B-1)/B ≤ q/B := Nat.le_of_not_gt h
    have mul := Nat.mul_le_mul_left B le
    omega
  refine ⟨q/B, ?_, ⟨⟨q%B,hm⟩,PUnit.unit⟩, ?_⟩
  · apply List.mem_flatMap.mpr
    refine ⟨r,List.mem_range.mpr hr,?_⟩
    apply List.mem_map.mpr
    exact ⟨q/B,List.mem_range.mpr hcover,rfl⟩
  · simpa only [Nat.mul_comm] using (Nat.div_add_mod q B).symm

/-- Any serial ordering of the source's full 2D launch, with all output cells
read from the same constructed final memory. No final-value premise. -/
theorem grid_result {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (B : Nat) (positive : 0 < B) (own : Ownership c)
    (tiles : List (Nat × Nat)) (schedule : tiles.Perm (grid c B)) (s : State α) :
    ∃ final, runTiles M c B tiles s = some final ∧
      (∀ r q, r < c.rows → q < c.cols →
        final.mem c.OUT (r*c.so+q) = Cell.mk .real (scalar M
          ((s.mem c.X (r*c.sx+q)).read .real) ((s.mem c.Y (r*c.sy+q)).read .real))) ∧
      (∀ region a, (region ≠ c.OUT ∨ ∀ r q, r < c.rows → q < c.cols → a ≠ r*c.so+q) →
        final.mem region a = s.mem region a) := by
  obtain ⟨final, ran, values, frame⟩ := tiles_result M c B own tiles
    (fun t ht => grid_valid c B t (schedule.mem_iff.mp ht)) s
  refine ⟨final,ran,?_,?_⟩
  · intro r q hr hq
    apply values r q hr hq
    obtain ⟨b,hb,lane,heq⟩ := grid_covers c B positive r q hr hq
    exact ⟨b,schedule.mem_iff.mpr hb,lane,heq⟩
  · intro region a outside
    apply frame
    intro t ht
    rcases outside with different | outside
    · exact Or.inl different
    · exact Or.inr (fun lane active => outside t.1 (t.2*B+lane.1.val)
        (grid_valid c B t (schedule.mem_iff.mp ht)) active)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SwigluFwd.Whole
