import bench.tritonbench_g.mean_reduction.Proofs.Mathematical_WholeFPValue

namespace VeriTile.Bench.TritonBenchG.MeanReduction.WholeFPFamily
open VeriTile.Triton FP.Structural WholeFPValue
noncomputable section
set_option maxHeartbeats 3000000

def runTiles {α : Type} [Inhabited α] (family : Nat → Algebra α) (variant : Bool)
    (X Mean : RegionName) (M N : Nat) : List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (exec (family pid) (typedKernel variant X Mean M N 8 8)
      (WholeFP.tileState pid s)).bind (runTiles family variant X Mean M N rest)

def rowValue {α : Type} [Inhabited α] (family : Nat → Algebra α) (X : RegionName)
    (M N : Nat) (memory : RegionName → Nat → Cell α) (row : Nat) : α :=
  WholeFPValue.rowValue (family (row/8)) X M N memory row

theorem rowValue_at {α : Type} [Inhabited α] (family : Nat → Algebra α) (X : RegionName)
    (M N pid : Nat) (memory : RegionName → Nat → Cell α) (i : Fin 8) :
    rowValue family X M N memory (pid*8+i.val)=tileValue (family pid) X M N pid memory i := by
  have hp : (pid*8+i.val)/8=pid := by omega
  simp only [rowValue,hp,WholeFPValue.rowValue_at]

theorem tiles_result {α : Type} [Inhabited α] (family : Nat → Algebra α) (X Mean : RegionName)
    (M N : Nat) (separate : X≠Mean) (tiles : List Nat) (s : State α) :
    ∃ final, runTiles family Bool.false X Mean M N tiles s=some final ∧
      (∀ i, i<M → Whole.Covered tiles i → final.mem Mean i=
        .mk .real (rowValue family X M N s.mem i)) ∧
      (∀ r o, (r≠Mean ∨ ∀ pid∈tiles, ∀ i : Fin 8,
        pid*8+i.val<M → o≠pid*8+i.val) → final.mem r o=s.mem r o) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · rintro i _ ⟨pid,hpid,_⟩; simp at hpid
    · intro r o _; rfl
  | cons pid rest ih =>
    obtain ⟨middle,ran,values,frame⟩ := tile_result (family pid) X Mean M N (WholeFP.tileState pid s)
    obtain ⟨final,tail,valuesTail,frameTail⟩ := ih middle
    have inputFrame (i : Nat) : (middle.mem X i).read .real=(s.mem X i).read .real := by
      rw [frame X i (Or.inl separate)]; rfl
    refine ⟨final,?_,?_,?_⟩
    · change (exec (family pid) (source X Mean M N) (WholeFP.tileState pid s)).bind _=_
      rw [ran]; exact tail
    · intro i hi covered
      by_cases hc : Whole.Covered rest i
      · rw [valuesTail i hi hc]
        congr 1
        exact tileValue_congr (family (i/8)) X M N (i/8) middle.mem s.mem inputFrame _
      · have preserve : final.mem Mean i=middle.mem Mean i := by
          apply frameTail Mean i
          right; intro q hq lane active equal
          exact hc ⟨q,hq,lane,equal⟩
        rw [preserve]
        rcases covered with ⟨q,hq,lane,equal⟩
        rcases List.mem_cons.mp hq with head | tail
        · subst q; subst i
          rw [rowValue_at]
          exact values lane hi
        · exact False.elim (hc ⟨q,tail,lane,equal⟩)
    · intro r o outside
      rw [frameTail r o (by
        rcases outside with hr | ho
        · exact Or.inl hr
        · exact Or.inr fun q hq lane active => ho q (List.mem_cons_of_mem pid hq) lane active)]
      apply frame r o
      rcases outside with hr | ho
      · exact Or.inl hr
      · exact Or.inr fun lane active => ho pid List.mem_cons_self lane active

theorem grid_result {α : Type} [Inhabited α] (family : Nat → Algebra α) (X Mean : RegionName)
    (M N : Nat) (separate : X≠Mean) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (Whole.gridSize M))) (s : State α) :
    ∃ final, runTiles family Bool.false X Mean M N tiles s=some final ∧
      (∀ i, i<M → final.mem Mean i=.mk .real (rowValue family X M N s.mem i)) ∧
      (∀ r o, (r≠Mean ∨ M ≤ o) → final.mem r o=s.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := tiles_result family X Mean M N separate tiles s
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    apply values i hi
    obtain ⟨pid,hpid,lane,equal⟩ := Whole.grid_covers M i hi
    exact ⟨pid,schedule.mem_iff.mpr hpid,lane,equal⟩
  · intro r o outside
    apply frame r o
    rcases outside with hr | ho
    · exact Or.inl hr
    · right; intro pid hpid lane active; omega

theorem grid_equal {α : Type} [Inhabited α] (family : Nat → Algebra α) (X Mean : RegionName)
    (M N : Nat) (tiles : List Nat) (s : State α) :
    runTiles family Bool.false X Mean M N tiles s=runTiles family Bool.true X Mean M N tiles s := by
  induction tiles generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [runTiles,WholeFP.tile_equal]
    cases exec (family pid) (typedKernel Bool.true X Mean M N 8 8) (WholeFP.tileState pid s) with
    | none => rfl
    | some t => exact ih t

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.WholeFPFamily
