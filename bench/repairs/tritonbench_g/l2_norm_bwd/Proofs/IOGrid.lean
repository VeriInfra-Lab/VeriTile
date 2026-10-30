import bench.repairs.tritonbench_g.l2_norm_bwd.IO
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.L2NormBwd.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def Outside (layout : Strides) (rows N : Nat) (name : RegionName) (offset : Nat) : Prop :=
  name≠"DX" ∨ ∀ row col,row<rows → col<N → offset≠row*layout.dxRow+col*layout.dxCol

theorem grid_execution {A : Type} [Inhabited A] (M : Algebra A) (layout : Strides)
    (rows N B : Nat) (cover : N≤B) (unique : OutputUnique layout rows N)
    (eps : A) (s : State A) :
    ∃ final,runSchedule (schedule M layout rows N B eps) s=some final ∧
      (∀ row col,row<rows → col<N →
        final.mem "DX" (row*layout.dxRow+col*layout.dxCol)=.mk .real
          (Tile.scalar M N B eps (Tile.input s "X" row layout.xRow layout.xCol)
            (Tile.input s "DY" row layout.dyRow layout.dyCol) col)) ∧
      (∀ name offset,Outside layout rows N name offset → final.mem name offset=s.mem name offset) := by
  induction rows with
  | zero => exact ⟨s,rfl,by intro _ _ h;omega,fun _ _ _ => rfl⟩
  | succ rows ih =>
    obtain ⟨previous,ran,values,frame⟩ := ih
      (fun a b c d ha hb hc hd h => unique a b c d (by omega) hb (by omega) hd h)
    let final := (Tile.sourceRun M layout N B rows eps previous).val
    have last : (invocation M layout N B rows eps).run previous=some final :=
      (Tile.sourceRun M layout N B rows eps previous).property
    refine ⟨final,?_,?_,?_⟩
    · simp only [schedule,List.range_succ,List.map_append,runSchedule_append,List.map_singleton]
      change (runSchedule (schedule M layout rows N B eps) s).bind _=some final
      rw [ran,Option.bind_some,runSchedule_singleton,last]
    · intro row col hr hc
      by_cases same : row=rows
      · subst row
        have hx : Tile.input previous "X" rows layout.xRow layout.xCol=
            Tile.input s "X" rows layout.xRow layout.xCol := by
          funext j
          exact congrArg (Cell.read .real) (frame "X" _ (Or.inl (by decide)))
        have hy : Tile.input previous "DY" rows layout.dyRow layout.dyCol=
            Tile.input s "DY" rows layout.dyRow layout.dyCol := by
          funext j
          exact congrArg (Cell.read .real) (frame "DY" _ (Or.inl (by decide)))
        rw [Tile.readback M layout N B rows eps previous
          (⟨col,lt_of_lt_of_le hc cover⟩,PUnit.unit) hc ?_,hx,hy]
        intro a b ha hb equal
        exact (unique rows a rows b (by omega) ha (by omega) hb
          (congrArg (rows*layout.dxRow+·) equal)).2
      · have earlier : row<rows := by omega
        rw [Tile.frame M layout N B rows eps previous "DX" _ (Or.inr ?_),values row col earlier hc]
        intro j hj equal
        exact same (unique row col rows j.1.val hr hc (by omega) hj equal).1
    · intro name offset outside
      have lastFrame : final.mem name offset=previous.mem name offset := by
        apply Tile.frame
        rcases outside with different | outside
        · exact Or.inl different
        · exact Or.inr (fun j hj => outside rows j.1.val (by omega) hj)
      rw [lastFrame]
      apply frame
      exact outside.imp id (fun h row col hr hc => h row col (by omega) hc)

theorem io_safe {A B : Type} [Inhabited B] (M : Algebra B)
    (encode : A → B) (observe : Nat → B → A) (layout : Strides)
    (rows N block : Nat) (eps : B) (xs : Inputs A rows N) (bounds : RegionName → Nat)
    (hb : (sourceIO M encode observe layout rows N block eps).InBounds bounds xs)
    (s : State B) : ScheduleSafe bounds (schedule M layout rows N block eps) s := by
  suffices ∀ jobs : List Nat,(∀ row∈jobs,row<rows) → ∀ initial,
      ScheduleSafe bounds (jobs.map (fun row => invocation M layout N block row eps)) initial from
    this _ (fun _ h => List.mem_range.mp h) s
  intro jobs
  induction jobs with
  | nil => intro _ _;trivial
  | cons row jobs ih =>
    intro members initial
    have hr := members row (by simp)
    refine ⟨?_,fun final _ => ih (fun a h => members a (by simp [h])) final⟩
    apply Tile.source_traceSafe
    · intro j hj
      exact hb.2.1 ⟨0,by simp [sourceIO,inputs]⟩ (⟨row,hr⟩,⟨j.val,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.1 ⟨1,by simp [sourceIO,inputs]⟩ (⟨row,hr⟩,⟨j.val,hj⟩,PUnit.unit) trivial
    · intro j hj
      exact hb.2.2.1 ⟨0,by simp [sourceIO]⟩ (⟨row,hr⟩,⟨j.val,hj⟩,PUnit.unit) trivial

theorem row_binding {A B : Type} [Inhabited A] [Inhabited B] (M : Algebra B)
    (encode : A → B) (observe : Nat → B → A) (layout : Strides)
    (rows N block : Nat) (eps : B) (xs : Inputs A rows N) (s : State B)
    (hin : (sourceIO M encode observe layout rows N block eps).InputBinding xs s)
    (row : Fin rows) (col : Nat) (hc : col<N) :
    Tile.input s "X" row.val layout.xRow layout.xCol col=encode (rowValues xs Bool.false row col) ∧
    Tile.input s "DY" row.val layout.dyRow layout.dyCol col=encode (rowValues xs Bool.true row col) := by
  constructor
  · have h := hin ⟨0,by simp [sourceIO,inputs]⟩ (row,⟨col,hc⟩,PUnit.unit) trivial
    change (s.mem _ _).read .real=encode _ at h
    simpa only [rowValues,hc,↓reduceDIte,↓reduceIte,Bool.false_eq_true] using h
  · have h := hin ⟨1,by simp [sourceIO,inputs]⟩ (row,⟨col,hc⟩,PUnit.unit) trivial
    change (s.mem _ _).read .real=encode _ at h
    simpa only [rowValues,hc,↓reduceDIte,↓reduceIte] using h

theorem outside_of_io {A B : Type} (M : Algebra B) (encode : A → B) (observe : Nat → B → A)
    (layout : Strides) (rows N block : Nat) (eps : B) (xs : Inputs A rows N)
    (name : RegionName) (offset : Nat)
    (h : (sourceIO M encode observe layout rows N block eps).Outside xs name offset) :
    Outside layout rows N name offset := by
  by_cases same : name="DX"
  · right
    intro row col hr hc
    exact (h.1 ⟨0,by simp [sourceIO]⟩ (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) trivial).resolve_left
      (not_not.mpr same)
  · exact Or.inl same

/-- Compose the source row certificate and full strided schedule with a scalar
result proved independently for the chosen Real or native backend. -/
theorem implements {A B : Type} [Inhabited A] [Inhabited B] (M : Algebra B)
    (encode : A → B) (observe : Nat → B → A) (layout : Strides)
    (rows N block : Nat) (cover : N≤block) (unique : OutputUnique layout rows N)
    (eps : B) (f : (Nat → A) → (Nat → A) → Nat → A)
    (formula : ∀ xs ys col,col<N →
      observe col (Tile.scalar M N block eps (encode ∘ xs) (encode ∘ ys) col)=f xs ys col) :
    (sourceIO M encode observe layout rows N block eps).Implements
      (fun xs => result rows N (fun row col =>
        f (rowValues xs Bool.false row) (rowValues xs Bool.true row) col.val)) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    exact io_safe M encode observe layout rows N block eps xs bounds hb initial
  · intro xs initial hin
    obtain ⟨final,ran,values,frame⟩ := grid_execution M layout rows N block cover unique eps initial
    refine ⟨final,ran,?_,?_⟩
    · intro port
      refine Fin.cases ?_ (fun k => k.elim0) port
      intro j _
      change mapCell (observe j.2.1.val)
        (final.mem "DX" (j.1.val*layout.dxRow+j.2.1.val*layout.dxCol))=Cell.mk .real _
      rw [values j.1.val j.2.1.val j.1.isLt j.2.1.isLt]
      change Cell.mk .real (observe j.2.1.val (Tile.scalar _ _ _ _ _ _ _))=Cell.mk .real _
      apply congrArg (Cell.mk .real)
      rw [Tile.scalar_congr M N block eps _ _
        (encode ∘ rowValues xs Bool.false j.1) (encode ∘ rowValues xs Bool.true j.1)
        (fun col hc => (row_binding M encode observe layout rows N block eps xs initial hin j.1 col hc).1)
        (fun col hc => (row_binding M encode observe layout rows N block eps xs initial hin j.1 col hc).2)
        j.2.1.val j.2.1.isLt]
      exact formula _ _ _ j.2.1.isLt
    · intro name offset outside
      exact frame name offset (outside_of_io M encode observe layout rows N block eps xs name offset outside)

#axiomsClean grid_execution
#axiomsClean implements
end
end VeriTile.Bench.Repairs.L2NormBwd.PublicIO
