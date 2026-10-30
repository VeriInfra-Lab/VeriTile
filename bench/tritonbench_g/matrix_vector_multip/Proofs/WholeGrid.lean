import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeSource

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeGrid
open VeriTile.Triton FP.Structural WholeSource
open Typed (Parameters)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def gridSize (p : Parameters) := (p.rows+p.blockRows-1)/p.blockRows
def tileState {α : Type} (pid : Nat) (s : State α) : State α :=
  {s with pids:=fun ax => if ax=0 then pid else s.pids ax}
def runTiles {α : Type} [Inhabited α] (family : Nat → Algebra α) (p : Parameters) :
    List Nat → State α → Option (State α)
  | [],s => some s
  | pid::rest,s => (exec (family pid) (source p) (tileState pid s)).bind (runTiles family p rest)
def Covered (p : Parameters) (tiles : List Nat) (i : Nat) : Prop :=
  ∃ pid∈tiles, ∃ lane : Fin p.blockRows, i=row p pid lane.val

theorem grid_covers (p : Parameters) (rows : 0<p.blockRows) (i : Nat) (hi : i<p.rows) :
    Covered p (List.range (gridSize p)) i := by
  have hg : i/p.blockRows<gridSize p := by
    apply Nat.lt_of_succ_le
    apply (Nat.le_div_iff_mul_le rows).mpr
    have h := Nat.div_mul_le_self i p.blockRows
    rw [Nat.succ_mul]
    omega
  refine ⟨i/p.blockRows,List.mem_range.mpr hg,⟨i%p.blockRows,Nat.mod_lt _ rows⟩,?_⟩
  simpa [row,Nat.mul_comm] using (Nat.div_add_mod i p.blockRows).symm

def rowValue {α : Type} [Inhabited α] (family : Nat → Algebra α) (p : Parameters)
    (rows : 0<p.blockRows) (memory : RegionName → Nat → Cell α) (r : Nat) : α :=
  tileValue (family (r/p.blockRows)) p (r/p.blockRows) memory ⟨r%p.blockRows,Nat.mod_lt _ rows⟩

theorem rowValue_at {α : Type} [Inhabited α] (family : Nat → Algebra α) (p : Parameters)
    (rows : 0<p.blockRows) (memory : RegionName → Nat → Cell α) (pid : Nat) (i : Fin p.blockRows) :
    rowValue family p rows memory (row p pid i.val)=tileValue (family pid) p pid memory i := by
  have hp : (row p pid i.val)/p.blockRows=pid := by
    unfold row
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ rows,Nat.div_eq_of_lt i.isLt,Nat.zero_add]
  have hi : (⟨(row p pid i.val)%p.blockRows,Nat.mod_lt _ rows⟩ : Fin p.blockRows)=i := by
    apply Fin.ext
    simp [row,Nat.add_mod,Nat.mod_eq_of_lt i.isLt]
  simp only [rowValue,hp,hi]

theorem accumulated_congr {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid : Nat) (memory other : RegionName → Nat → Cell α)
    (sameA : ∀ i,(memory p.a i).read .fp32=(other p.a i).read .fp32)
    (sameB : ∀ i,(memory p.b i).read .fp32=(other p.b i).read .fp32)
    (off : Nat) (acc : Acc α p) :
    accumulated A p pid memory off acc=accumulated A p pid other off acc := by
  conv_lhs => rw [accumulated]
  conv_rhs => rw [accumulated]
  split
  · have next : nextAcc A p pid off memory acc=nextAcc A p pid off other acc := by
      funext i
      simp only [nextAcc,aChunk,bChunk,load,sameA,sameB]
    rw [next]
    exact accumulated_congr A p pid memory other sameA sameB (off+p.blockColumns) _
  · rfl
termination_by p.columns-off

theorem tileValue_congr {α : Type} [Inhabited α] (A : Algebra α) (p : Parameters)
    (pid : Nat) (memory other : RegionName → Nat → Cell α)
    (sameA : ∀ i,(memory p.a i).read .fp32=(other p.a i).read .fp32)
    (sameB : ∀ i,(memory p.b i).read .fp32=(other p.b i).read .fp32)
    (i : Fin p.blockRows) : tileValue A p pid memory i=tileValue A p pid other i := by
  unfold tileValue
  rw [accumulated_congr A p pid memory other sameA sameB]

theorem tiles_result {α : Type} [Inhabited α] (family : Nat → Algebra α) (p : Parameters)
    (rows : 0<p.blockRows) (width : 0<p.blockColumns) (stride : p.strideOut=1)
    (separateA : p.a≠p.out) (separateB : p.b≠p.out) (tiles : List Nat) (s : State α) :
    ∃ final, runTiles family p tiles s=some final ∧
      (∀ i,i<p.rows → Covered p tiles i → final.mem p.out i=
        .mk .fp32 (rowValue family p rows s.mem i)) ∧
      (∀ r o,(r≠p.out ∨ ∀ pid∈tiles,∀ i : Fin p.blockRows,
        row p pid i.val<p.rows → o≠row p pid i.val) → final.mem r o=s.mem r o) := by
  induction tiles generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · rintro i _ ⟨pid,hpid,_⟩; simp at hpid
    · intro r o _; rfl
  | cons pid rest ih =>
    obtain ⟨middle,ran,values,frame⟩ := tile_result (family pid) p width stride (tileState pid s)
    obtain ⟨final,tailRan,tailValues,tailFrame⟩ := ih middle
    have sameA (i : Nat) : (middle.mem p.a i).read .fp32=(s.mem p.a i).read .fp32 := by
      rw [frame p.a i (Or.inl separateA)]; rfl
    have sameB (i : Nat) : (middle.mem p.b i).read .fp32=(s.mem p.b i).read .fp32 := by
      rw [frame p.b i (Or.inl separateB)]; rfl
    refine ⟨final,?_,?_,?_⟩
    · change (exec (family pid) (source p) (tileState pid s)).bind _=_
      rw [ran]
      exact tailRan
    · intro i hi covered
      by_cases hc : Covered p rest i
      · rw [tailValues i hi hc]
        congr 1
        exact tileValue_congr _ p _ middle.mem s.mem sameA sameB _
      · have preserve : final.mem p.out i=middle.mem p.out i := by
          apply tailFrame p.out i
          right
          intro q hq lane active equal
          exact hc ⟨q,hq,lane,equal⟩
        rw [preserve]
        rcases covered with ⟨q,hq,lane,equal⟩
        rcases List.mem_cons.mp hq with head | inTail
        · subst q; subst i
          rw [rowValue_at]
          exact values lane hi
        · exact False.elim (hc ⟨q,inTail,lane,equal⟩)
    · intro r o outside
      rw [tailFrame r o (by
        rcases outside with hr | ho
        · exact Or.inl hr
        · exact Or.inr fun q hq lane active => ho q (List.mem_cons_of_mem pid hq) lane active)]
      apply frame r o
      rcases outside with hr | ho
      · exact Or.inl hr
      · exact Or.inr fun lane active => ho pid List.mem_cons_self lane active

/-- Complete actual source grid, in any legal block order, including empty
grids, every tail, original input reads and the full exterior memory frame. -/
theorem grid_result {α : Type} [Inhabited α] (family : Nat → Algebra α) (p : Parameters)
    (rows : 0<p.blockRows) (width : 0<p.blockColumns) (stride : p.strideOut=1)
    (separateA : p.a≠p.out) (separateB : p.b≠p.out) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (gridSize p))) (s : State α) :
    ∃ final,runTiles family p tiles s=some final ∧
      (∀ i,i<p.rows → final.mem p.out i=.mk .fp32 (rowValue family p rows s.mem i)) ∧
      (∀ r o,(r≠p.out ∨ p.rows≤o) → final.mem r o=s.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := tiles_result family p rows width stride separateA separateB tiles s
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    apply values i hi
    obtain ⟨pid,hpid,lane,equal⟩ := grid_covers p rows i hi
    exact ⟨pid,schedule.mem_iff.mpr hpid,lane,equal⟩
  · intro r o outside
    apply frame r o
    rcases outside with hr | ho
    · exact Or.inl hr
    · right; intro pid hp lane active; omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeGrid
