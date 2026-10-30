import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Tile

namespace VeriTile.Bench.Repairs.SwigluTriton.Grid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

theorem row_unique (block r c r' c' : Nat) (hc : c < block) (hc' : c' < block)
    (equal : r*block+c=r'*block+c') : r=r' ∧ c=c' := by
  have forward : r≤r' := by
    by_contra h
    have hmul := Nat.mul_le_mul_right block (show r'+1≤r by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hmul
    omega
  have backward : r'≤r := by
    by_contra h
    have hmul := Nat.mul_le_mul_right block (show r+1≤r' by omega)
    simp only [Nat.add_mul, Nat.one_mul] at hmul
    omega
  have same : r=r' := by omega
  exact ⟨same, by subst r'; omega⟩

def runRows {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool) (block cols : Nat) :
    List Nat → State α → Option (State α)
  | [], s => some s
  | row :: rest, s =>
      (run M (source backward block cols).surfaceBody (Tile.rowState row s)).bind
        (runRows M backward block cols rest)

theorem row_frame {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols written observed col : Nat) (s : State α) (region : RegionName)
    (bounded : cols≤block) (active : col<cols) (different : observed≠written) :
    (Tile.sourceRun M backward block cols written s).val.mem region (observed*block+col)=
      s.mem region (observed*block+col) := by
  apply Tile.frame
  intro ch on
  right
  intro i hi equal
  exact different (row_unique block observed col written i.1.val
    (lt_of_lt_of_le active bounded) i.1.isLt equal).1

/-- No row is repeated: in-place backward reads that row's entry A/B before
either store, while every other program acts on disjoint padded row words.
The schedule can be any permutation of the grid, not only increasing order. -/
theorem rows_result {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (block cols : Nat) (bounded : cols≤block) (rows : List Nat) (unique : rows.Nodup)
    (s : State α) :
    ∃ final, runRows M backward block cols rows s=some final ∧
      (∀ ch, enabled backward ch → ∀ row col, row∈rows → col<cols →
        final.mem (outputRegion ch) (row*block+col)=Cell.mk .real
          (Tile.scalar M ch ((s.mem "A" (row*block+col)).read .real)
            ((s.mem "B" (row*block+col)).read .real)
            ((s.mem "DC" (row*block+col)).read .real))) ∧
      (∀ region address, (∀ row∈rows,Tile.Outside backward block cols row region address) →
        final.mem region address=s.mem region address) := by
  induction rows generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro ch on row col member; simp at member
    · intro region address outside; rfl
  | cons head rest ih =>
    obtain ⟨notMember,uniqueRest⟩ := List.nodup_cons.mp unique
    let middle := (Tile.sourceRun M backward block cols head s).val
    obtain ⟨final,ran,values,frame⟩ := ih uniqueRest middle
    refine ⟨final,?_,?_,?_⟩
    · simp only [runRows,(Tile.sourceRun M backward block cols head s).property,
        Option.bind_some]
      exact ran
    · intro ch on row col member active
      rcases List.mem_cons.mp member with same | tail
      · subst row
        have keep : final.mem (outputRegion ch) (head*block+col)=
            middle.mem (outputRegion ch) (head*block+col) := by
          apply frame
          intro r hr other otherOn
          right
          intro lane hl equal
          have same := (row_unique block head col r lane.1.val
            (lt_of_lt_of_le active bounded) lane.1.isLt equal).1
          exact notMember (same ▸ hr)
        rw [keep]
        exact Tile.readback M backward block cols head s ch on
          (⟨⟨col,lt_of_lt_of_le active bounded⟩,PUnit.unit⟩) active
      · have different : row≠head := fun same => notMember (same ▸ tail)
        rw [values ch on row col tail active]
        rw [row_frame M backward block cols head row col s "A" bounded active different,
          row_frame M backward block cols head row col s "B" bounded active different,
          row_frame M backward block cols head row col s "DC" bounded active different]
    · intro region address outside
      rw [frame region address (fun row hr => outside row (List.mem_cons_of_mem head hr))]
      exact Tile.frame M backward block cols head s region address
        (outside head List.mem_cons_self)

def Outside (backward : Bool) (rows cols block : Nat) (region : RegionName) (address : Nat) : Prop :=
  ∀ ch,enabled backward ch → region≠outputRegion ch ∨
    ∀ row col,row<rows → col<cols → address≠row*block+col

theorem whole {α : Type} [Inhabited α] (M : Algebra α) (backward : Bool)
    (rows cols block : Nat) (bounded : cols≤block) (schedule : List Nat)
    (allRows : schedule.Perm (List.range rows)) (s : State α) :
    ∃ final,runRows M backward block cols schedule s=some final ∧
      (∀ ch,enabled backward ch → ∀ row col,row<rows → col<cols →
        final.mem (outputRegion ch) (row*block+col)=Cell.mk .real
          (Tile.scalar M ch ((s.mem "A" (row*block+col)).read .real)
            ((s.mem "B" (row*block+col)).read .real)
            ((s.mem "DC" (row*block+col)).read .real))) ∧
      (∀ region address,Outside backward rows cols block region address →
        final.mem region address=s.mem region address) := by
  obtain ⟨final,ran,values,frame⟩ := rows_result M backward block cols bounded schedule
    (allRows.nodup_iff.mpr List.nodup_range) s
  refine ⟨final,ran,?_,?_⟩
  · intro ch on row col hr hc
    exact values ch on row col (allRows.mem_iff.mpr (List.mem_range.mpr hr)) hc
  · intro region address outside
    apply frame
    intro row member ch on
    rcases outside ch on with separate | away
    · exact Or.inl separate
    · exact Or.inr (fun i hi => away row i.1.val
        (List.mem_range.mp (allRows.mem_iff.mp member)) hi)

#axiomsClean rows_result
#axiomsClean whole
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Grid
