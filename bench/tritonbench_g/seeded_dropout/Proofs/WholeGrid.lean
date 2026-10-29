import bench.tritonbench_g.seeded_dropout.Proofs.WholeTile

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

/-- Only block scheduling is chosen; every execution is the original source. -/
def tileExec {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) (pid : Nat)
    (s : State α) : Option (State α) :=
  run M (c.source kind p seed).surfaceBody (tileState pid s)

def runTiles {α : Type} [Inhabited α] (M : Algebra α) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind) :
    List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (tileExec M c kind p seed pid s).bind (runTiles M c kind p seed rest)

def Covered (c : Layout) (tiles : List Nat) (i : Nat) : Prop :=
  ∃ pid, pid ∈ tiles ∧ ∃ lane : TileIndex [c.B], i = pid*c.B+lane.1.val

theorem tiles_result {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind)
    (hn : c.n < 2^31) (hs : kind.inRange (seedInt kind seed) = Bool.true)
    (separate : c.X ≠ c.OUT) (tiles : List Nat)
    (valid : ∀ pid ∈ tiles, pid < SourceDomain.grid c.n c.B) (s : State α) :
    ∃ final, runTiles M c kind p seed tiles s = some final ∧
      (∀ i, i < c.n → Covered c tiles i → final.mem c.OUT i =
        Cell.mk .real (scalar M lt p (seedInt kind seed) i ((s.mem c.X i).read .real))) ∧
      (∀ region a, (∀ pid ∈ tiles, Outside c pid region a) → final.mem region a = s.mem region a) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro i hi covered
      rcases covered with ⟨pid,hpid,_⟩
      simp at hpid
    · intro region a _; rfl
  | cons pid rest ih =>
    have hp := valid pid List.mem_cons_self
    let middle := (sourceRun M lt compare c kind p seed pid s hn hp hs).val
    have ran : tileExec M c kind p seed pid s = some middle :=
      (sourceRun M lt compare c kind p seed pid s hn hp hs).property
    obtain ⟨final,tail,values,frame⟩ := ih (fun q hq => valid q (List.mem_cons_of_mem pid hq)) middle
    have xf (i : Nat) : middle.mem c.X i = s.mem c.X i :=
      tile_frame M lt compare c kind p seed pid s hn hp hs c.X i (Or.inl separate)
    refine ⟨final,?_,?_,?_⟩
    · simp only [runTiles,ran,Option.bind_some,tail]
    · intro i hi covered
      by_cases hc : Covered c rest i
      · rw [values i hi hc,xf]
      · have preserve : final.mem c.OUT i = middle.mem c.OUT i := by
          apply frame
          intro q hq
          right
          intro lane active equal
          exact hc ⟨q,hq,lane,equal⟩
        rw [preserve]
        rcases covered with ⟨q,hq,lane,equal⟩
        rcases List.mem_cons.mp hq with head | tail
        · subst q
          subst i
          exact tile_readback M lt compare c kind p seed pid s hn hp hs lane hi
        · exact False.elim (hc ⟨q,tail,lane,equal⟩)
    · intro region a outside
      rw [frame region a (fun q hq => outside q (List.mem_cons_of_mem pid hq))]
      exact tile_frame M lt compare c kind p seed pid s hn hp hs region a
        (outside pid List.mem_cons_self)

def grid (c : Layout) : List Nat := List.range (SourceDomain.grid c.n c.B)

theorem grid_covers (c : Layout) (i : Nat) (hi : i < c.n) : Covered c (grid c) i := by
  have positive := (SourceDomain.legalBlock_fields c.legal).1
  have bound := Index.quotient_lt_grid positive hi
  exact ⟨i/c.B,List.mem_range.mpr bound,⟨⟨i%c.B,Nat.mod_lt i positive⟩,PUnit.unit⟩,
    by simpa only [Nat.mul_comm] using (Nat.div_add_mod i c.B).symm⟩

/-- Complete c.B-wide source launch in any serial block order, including the
empty grid. Ownership and pairwise-distinct active writes justify scheduling. -/
theorem grid_result {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (compare : M.compareLt none .real = some lt) (c : Layout)
    (kind : Random.IntegerType) (p : ℝ) (seed : SeedCarrier kind)
    (hn : c.n < 2^31) (hs : kind.inRange (seedInt kind seed) = Bool.true)
    (separate : c.X ≠ c.OUT) (tiles : List Nat) (schedule : tiles.Perm (grid c)) (s : State α) :
    ∃ final, runTiles M c kind p seed tiles s = some final ∧
      (∀ i, i < c.n → final.mem c.OUT i =
        Cell.mk .real (scalar M lt p (seedInt kind seed) i ((s.mem c.X i).read .real))) ∧
      (∀ region a, (region ≠ c.OUT ∨ c.n ≤ a) → final.mem region a = s.mem region a) := by
  obtain ⟨final,ran,values,frame⟩ := tiles_result M lt compare c kind p seed hn hs separate tiles
    (fun pid hp => List.mem_range.mp (schedule.mem_iff.mp hp)) s
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    apply values i hi
    obtain ⟨pid,hp,lane,he⟩ := grid_covers c i hi
    exact ⟨pid,schedule.mem_iff.mpr hp,lane,he⟩
  · intro region a outside
    apply frame
    intro pid hp
    rcases outside with hr | ha
    · exact Or.inl hr
    · right; intro lane active equal; omega

theorem active_addresses_unique (c : Layout) (pid q : Nat) (i j : Fin c.B)
    (equal : pid*c.B+i.val = q*c.B+j.val) : pid = q ∧ i = j :=
  Index.offset_pair_unique (SourceDomain.legalBlock_fields c.legal).1 i j equal

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
