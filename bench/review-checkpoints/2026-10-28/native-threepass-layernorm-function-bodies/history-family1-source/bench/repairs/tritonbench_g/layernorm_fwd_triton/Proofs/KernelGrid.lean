import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.SourceContract
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.KernelGraph
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Address

namespace VeriTile.Bench.Repairs.LayernormFwdTriton.KernelGrid
open VeriTile.Triton FP.Structural Address
noncomputable section
set_option maxHeartbeats 2000000

variable {α : Type} [Inhabited α] {shape : Shape} {st : Strides}

def xValues (s : State α) (st : Strides) (row : Row shape) (col : Nat) : α :=
  (s.mem "X" (xOffset st row col)).read .fp32
def wValues (s : State α) (st : Strides) (row : Row shape) (col : Nat) : α :=
  (s.mem "W" (wOffset st row col)).read .fp32

def Outside (st : Strides) (rows : List (Row shape)) (region : RegionName) (offset : Nat) : Prop :=
  region≠"Y" ∨ ∀ row, row∈rows → ∀ col : Fin shape.columns, offset≠yOffset st row col.val

structure Effect (M : Algebra α) (st : Strides) (eps : α)
    (rows : List (Row shape)) (initial final : State α) : Prop where
  output : ∀ row, row∈rows → ∀ col : Fin shape.columns,
    final.mem "Y" (yOffset st row col.val)=.mk .fp32
      (Graph.output M shape.columns (xValues initial st row) (wValues initial st row) eps col.val)
  frame : ∀ region offset, Outside st rows region offset → final.mem region offset=initial.mem region offset

private theorem injective_columns (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) (row : Row shape) :
    ∀ a b, a<shape.columns → b<shape.columns → a*st.yDim=b*st.yDim → a=b := by
  intro a b ha hb he
  have h := injective (a₁:=(row,⟨a,ha⟩)) (a₂:=(row,⟨b,hb⟩))
    (by simpa [yOffset,offset,he])
  exact congrArg (fun q => q.2.val) h

/-- Actual source execution for one complete strided row, every output, and
all regional memory outside that row. -/
theorem row_complete (M : Algebra α) (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) (eps : α)
    (s : State α) (row : Row shape) :
    ∃ t, FP.Structural.exec M (Source.kernel (Source.parameters st shape.columns))
        (Source.atRow (s.setReg "eps" .real [] (fun _ => eps)) row)=some t ∧ Effect M st eps [row] s t := by
  let p := Source.parameters st shape.columns
  obtain ⟨t,run,mem⟩ := Source.complete_run M p (Source.atRow (s.setReg "eps" .real [] (fun _ => eps)) row) eps (by simp [Source.atRow]) (by simp [p,Source.parameters]) (by simp [p,Source.parameters])
  refine ⟨t,run,?_,?_⟩
  · intro q hq col
    have heq : q=row := by simpa using hq
    subst q
    rw [mem]
    have hr := Source.reference_readback M p (Source.atRow (s.setReg "eps" .real [] (fun _ => eps)) row) (by simp [p,Source.parameters])
      col.val col.isLt (injective_columns injective row)
    have hg := Graph.source_output M p (Source.atRow (s.setReg "eps" .real [] (fun _ => eps)) row) rfl
      (xValues s st row) (wValues s st row) (by
        intro j _; simp [p,Source.parameters,Source.xBase,Source.atRow,xValues,xOffset,offset]) (by
        intro j _; simp [p,Source.parameters,Source.wBase,Source.atRow,wValues,wOffset,offset]) col.val col.isLt
    rw [hg] at hr
    simpa [p,Source.parameters,Source.yBase,Source.atRow,Source.epsilonValue,State.setReg,yOffset,offset] using hr
  · intro r o outside
    rw [mem]
    apply Source.reference_frame M p (Source.atRow (s.setReg "eps" .real [] (fun _ => eps)) row) r o
    rcases outside with h|h
    · exact .inl h
    · right
      intro n hn
      simpa [p,Source.parameters,Source.yBase,Source.atRow,Source.epsilonValue,State.setReg,yOffset,offset] using
        h row (by simp) ⟨n,hn⟩

private theorem effect_inputs (M : Algebra α) (eps : α) (rows : List (Row shape))
    (s t : State α) (h : Effect M st eps rows s t) (row : Row shape) :
    xValues t st row=xValues s st row ∧ wValues t st row=wValues s st row := by
  constructor <;> funext col
  · simp only [xValues,h.frame "X" _ (.inl (by decide))]
  · simp only [wValues,h.frame "W" _ (.inl (by decide))]

theorem rows_complete (M : Algebra α) (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) (eps : α)
    (rows : List (Row shape)) (s : State α) :
    ∃ t, Source.runRows M (Source.parameters st shape.columns) eps rows s=some t ∧
      Effect M st eps rows s t := by
  induction rows generalizing s with
  | nil => exact ⟨s,rfl,⟨by simp,fun _ _ _ => rfl⟩⟩
  | cons row rows ih =>
    obtain ⟨a,ha,ea⟩ := row_complete M injective eps s row
    obtain ⟨t,ht,et⟩ := ih a
    refine ⟨t,?_,?_,?_⟩
    · simp only [Source.runRows,ha,bind,Option.bind,ht]
    · intro q hq col
      obtain ⟨hx,hw⟩ := effect_inputs M eps [row] s a ea q
      by_cases htail : q∈rows
      · simpa only [hx,hw] using et.output q htail col
      · have heq : q=row := by simpa [htail] using hq
        subst q
        rw [et.frame "Y" _ (.inr ?_)]
        · exact ea.output row (by simp) col
        · intro other ho j he
          have hsame := injective (a₁:=(row,col)) (a₂:=(other,j)) he
          have hrow := congrArg Prod.fst hsame
          change row=other at hrow
          exact htail (hrow.symm ▸ ho)
    · intro r o outside
      have tail : Outside st rows r o := by
        rcases outside with h|h
        · exact .inl h
        · exact .inr (fun q hq j => h q (List.mem_cons_of_mem _ hq) j)
      have head : Outside st [row] r o := by
        rcases outside with h|h
        · exact .inl h
        · right
          intro q hq j
          have heq : q=row := by simpa using hq
          subst q
          exact h row (by simp) j
      exact (et.frame r o tail).trans (ea.frame r o head)

/-- Successful typed source execution of the entire original logical grid,
all outputs, and the complete exterior regional frame. Empty axes need no
special positive-size contract or successful-run hypothesis. -/
theorem complete (M : Algebra α) (injective : Function.Injective (fun q : Coordinate shape => yOffset st q.1 q.2.val)) (eps : α) (s : State α) :
    ∃ t, Source.launch M shape st eps s=some t ∧
      Effect M st eps (Source.programs shape) s t :=
  rows_complete M injective eps (Source.programs shape) s

theorem mem_programs (row : Row shape) : row∈Source.programs shape := by
  simp [Source.programs]

#axiomsClean row_complete
#axiomsClean complete
end
end VeriTile.Bench.Repairs.LayernormFwdTriton.KernelGrid
