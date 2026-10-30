import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.nested_loops_processing.IO
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.LogicalReader
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 32768
/-- Signed-int32 arange input and zero initial output, as logical matrices. -/
def initialized (rows cols : Nat) (xs : LogicalView.Inputs rows cols) : Prop :=
  (∀ r c, xs.input r c=(BitVec.ofNat 32 (r.val*cols+c.val)).toInt) ∧
  (∀ r c, xs.initialOutput r c=0)

/-- Ordered 2×2 shifted copies. A column crossing the row width carries to
the next row. Stores are traversed in source order, so the final writer wins. -/
def integerTarget (rows cols : Nat) (row : Fin rows) (column : Fin cols) : Int :=
  let inputShift := fun t =>
    let i := t/12
    let j := t%12/6
    let k := t%6/3
    if t%3=0 then 14*i else if t%3=1 then 14*i+6*j+2 else 14*i+6*j+2*k+4
  (List.range 24).foldl (fun value t =>
    (List.range 2).foldl (fun value r =>
      (List.range 2).foldl (fun value c =>
        if row.val=r+(c+2*t)/cols ∧ column.val=(c+2*t)%cols then
          let inputRow := r+(c+inputShift t)/cols
          let inputColumn := (c+inputShift t)%cols
          if inputRow<rows then (BitVec.ofNat 32 (inputRow*cols+inputColumn)).toInt else 0
        else value) value) value) 0

/-- Evaluation commutes with a left fold whenever each complete step does. -/
theorem fold_evaluate {A S T : Type} (xs : List A) (f : S → A → S) (g : T → A → T)
    (eval : S → T) (step : ∀ s a, eval (f s a)=g (eval s) a) (s : S) :
    eval (xs.foldl f s)=xs.foldl g (eval s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons a xs ih => simpa only [List.foldl_cons,step] using ih (f s a)

theorem offset_eq (cols shift row col : Nat) :
    ((row : Int)*(cols : Int)+((col+shift : Nat) : Int)*1).toNat=row*cols+col+shift := by
  have h : (row : Int)*(cols : Int)+((col+shift : Nat) : Int)*1=((row*cols+col+shift : Nat) : Int) := by push_cast; ring
  rw [h,Int.toNat_natCast]

theorem coordinates_eq (cols r c row column : Nat) (positive : 0<cols) (columnBound : column<cols) :
    r*cols+c=row*cols+column ↔ row=r+c/cols ∧ column=c%cols := by
  constructor
  · intro eq
    have div := congrArg (fun n => n/cols) eq
    have mod := congrArg (fun n => n%cols) eq
    dsimp only at div mod
    have d1 : (r*cols+c)/cols=r+c/cols := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive]; omega
    have d2 : (row*cols+column)/cols=row := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt columnBound]; omega
    rw [d1,d2] at div
    simp [Nat.add_mod,Nat.mod_eq_of_lt columnBound] at mod
    exact ⟨div.symm,mod.symm⟩
  · rintro ⟨rfl,rfl⟩
    have := Nat.div_add_mod c cols
    nlinarith

theorem formula_eq (rows cols : Nat) (positive : 0<rows*cols) (row : Fin rows) (column : Fin cols) :
    integerTarget rows cols row column=ReaderBefore.integerTarget rows cols (row.val*cols+column.val) := by
  have hc : 0<cols := Nat.pos_of_mul_pos_left positive
  dsimp only [integerTarget,ReaderBefore.integerTarget]
  simp only [bind_pure_comp,List.map_eq_map,List.foldl_map]
  symm
  apply fold_evaluate (eval:=fun mem : Nat → Int => mem (row.val*cols+column.val))
  intro mem t
  apply fold_evaluate (eval:=fun mem : Nat → Int => mem (row.val*cols+column.val))
  intro mem r
  apply fold_evaluate (eval:=fun mem : Nat → Int => mem (row.val*cols+column.val))
  intro mem c
  simp only [Function.update_apply,offset_eq]
  have hit := coordinates_eq cols r (c+2*t) row.val column.val hc column.isLt
  have source : (r+(c+(if t%3=0 then 14*(t/12) else if t%3=1 then 14*(t/12)+6*(t%12/6)+2 else 14*(t/12)+6*(t%12/6)+2*(t%6/3)+4))/cols)*cols+
      (c+(if t%3=0 then 14*(t/12) else if t%3=1 then 14*(t/12)+6*(t%12/6)+2 else 14*(t/12)+6*(t%12/6)+2*(t%6/3)+4))%cols =
      r*cols+c+(if t%3=0 then 14*(t/12) else if t%3=1 then 14*(t/12)+6*(t%12/6)+2 else 14*(t/12)+6*(t%12/6)+2*(t%6/3)+4) := by
    have := Nat.div_add_mod (c+(if t%3=0 then 14*(t/12) else if t%3=1 then 14*(t/12)+6*(t%12/6)+2 else 14*(t/12)+6*(t%12/6)+2*(t%6/3)+4)) cols
    nlinarith
  rw [source]
  have bound (z : Nat) : r*cols+z<rows*cols ↔ r+z/cols<rows := by
    rw [←Nat.div_lt_iff_lt_mul hc]
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hc]
    omega
  simp only [Nat.add_assoc] at hit ⊢
  have hit' : row.val*cols+column.val=r*cols+(c+2*t) ↔
      row.val=r+(c+2*t)/cols ∧ column.val=(c+2*t)%cols := eq_comm.trans hit
  simp only [hit',bound]

/-- Complete exact-cell old/new contract, including both initialized snapshots. -/
theorem judgment_iff {α : Type} [Inhabited α] (M : Algebra α) (rows cols : Nat) (positive : 0<rows*cols) :
    (LogicalView.nestedIO M rows cols).ImplementsOn (initialized rows cols) (fun _ => integerTarget rows cols) ↔
    (ReaderBefore.nestedIO M rows cols).ImplementsOnCells (IO.initialized rows cols)
      (IO.result rows cols (ReaderBefore.integerTarget rows cols)) := by
  unfold LogicalDomainIO.ImplementsOn LogicalDomainIO.lift
  apply Iff.of_eq
  congr 1
  · funext xs
    apply propext
    change initialized rows cols (LogicalView.inputEquiv rows cols xs) ↔ IO.initialized rows cols xs
    unfold initialized IO.initialized
    constructor
    · rintro ⟨hin,hout⟩
      constructor
      · intro j
        obtain ⟨⟨r,c⟩,eq⟩ := finProdFinEquiv.surjective j.1
        have hj : j=(finProdFinEquiv (r,c),()) := by rw [eq]; rfl
        rw [hj]
        have := hin r c
        simpa [LogicalView.inputEquiv,LogicalView.matrixEquiv,←eq,finProdFinEquiv,signedValue,arangeWord,Nat.mul_comm,Nat.add_comm] using this
      · intro j
        obtain ⟨⟨r,c⟩,eq⟩ := finProdFinEquiv.surjective j.1
        have hj : j=(finProdFinEquiv (r,c),()) := by rw [eq]; rfl
        rw [hj]
        simpa [LogicalView.inputEquiv,LogicalView.matrixEquiv] using hout r c
    · rintro ⟨hin,hout⟩
      constructor
      · intro r c
        simpa [LogicalView.inputEquiv,LogicalView.matrixEquiv,finProdFinEquiv,signedValue,arangeWord,Nat.mul_comm,Nat.add_comm] using hin (finProdFinEquiv (r,c),())
      · intro r c
        exact hout (finProdFinEquiv (r,c),())
  · funext xs o j
    have ho : o=0 := Fin.eq_zero o
    subst o
    change integerTarget rows cols (finProdFinEquiv.symm j.1).1 (finProdFinEquiv.symm j.1).2 = ReaderBefore.integerTarget rows cols j.1.val
    rw [formula_eq rows cols positive]
    congr 1
    have eq := congrArg Fin.val (finProdFinEquiv.apply_symm_apply j.1)
    change (finProdFinEquiv.symm j.1).2.val+cols*(finProdFinEquiv.symm j.1).1.val=j.1.val at eq
    simpa only [Nat.mul_comm,Nat.add_comm] using eq
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.LogicalReader
