import bench.tritonbench_g.matmul_dequant_int4.Proofs.Source
import bench.tritonbench_g.matmul_dequant_int4.Sources.Grid
import bench.tritonbench_g.int_scaled_matmul.Sources.GridExecution

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.KernelGrid
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 4000000

abbrev Coordinate (p : Parameters) := Fin p.rows × Fin p.columns

def address (p : Parameters) (x : Coordinate p) : Nat := x.1.val*p.columns+x.2.val

def atProgram {α : Type} (p : Parameters) (pid : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then pid/(MatmulDequantInt4.ceil p.columns p.blockColumns)
    else if ax=1 then pid%(MatmulDequantInt4.ceil p.columns p.blockColumns) else 0}

def count (p : Parameters) :=
  MatmulDequantInt4.ceil p.rows p.blockRows * MatmulDequantInt4.ceil p.columns p.blockColumns

def one {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters)
    (pid : Nat) (s : State α) := FP.Structural.exec M (source f p) (atProgram p pid s)

def run {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters) (s : State α) :=
  GridExecution.sequence (one f M p) (List.range (count p)) s

/-- Independent instruction expression at logical matrix coordinates. -/
def result {α : Type} [Inhabited α] (f : Profile) (M : Algebra α) (p : Parameters)
    (s : State α) (x : Coordinate p) : α :=
  M.binary none f.dtype .mul
    (M.cast (some .fp32) .real f.dtype (M.fromInt (some .fp32)
      (nibble ((s.mem p.weight (x.1.val/8*p.weightRowStride+x.2.val*p.weightColumnStride)).read .int) x.1.val -
       nibble ((s.mem p.zero (x.1.val/p.groupSize*p.zeroRowStride+x.2.val/8*p.zeroColumnStride)).read .int) x.2.val)))
    (match f with
    | .half => (s.mem p.scale (x.1.val/p.groupSize*p.scaleRowStride+x.2.val*p.scaleColumnStride)).read .fp16
    | .single => ((s.mem p.scale (x.1.val/p.groupSize*p.scaleRowStride+x.2.val*p.scaleColumnStride)).read .fp32))

structure Valid (p : Parameters) : Prop where
  br : 0<p.blockRows
  bc : 0<p.blockColumns
  outputRow : p.outputRowStride=p.columns
  outputColumn : p.outputColumnStride=1
  weight : p.weight≠p.output
  zero : p.zero≠p.output
  scale : p.scale≠p.output

def owns (p : Parameters) (pid : Nat) (x : Coordinate p) : Prop :=
  ∃ i : Index p, pid/(MatmulDequantInt4.ceil p.columns p.blockColumns)*p.blockRows+i.1.val=x.1.val ∧
    pid%(MatmulDequantInt4.ceil p.columns p.blockColumns)*p.blockColumns+i.2.1.val=x.2.val

theorem address_injective (p : Parameters) : Function.Injective (address p) := by
  intro x y h
  have hh := MatmulDequantInt4.address_injective p.columns ⟨x.1,x.2⟩ ⟨y.1,y.2⟩ x.2.isLt y.2.isLt h
  exact Prod.ext (Fin.ext (congrArg MatmulDequantInt4.Work.row hh))
    (Fin.ext (congrArg MatmulDequantInt4.Work.col hh))

theorem local_injective {α : Type} (p : Parameters) (h : Valid p) (s : State α)
    (i j : Index p) (hi : active p s i) (hj : active p s j)
    (eq : outputOffset p s i=outputOffset p s j) : i=j := by
  have hh := MatmulDequantInt4.address_injective p.columns
    ⟨row p s i,column p s i⟩ ⟨row p s j,column p s j⟩ hi.2 hj.2
    (by simpa [outputOffset,h.outputRow,h.outputColumn] using eq)
  have hr := congrArg MatmulDequantInt4.Work.row hh
  have hc := congrArg MatmulDequantInt4.Work.col hh
  simp only [row,column] at hr hc
  apply Prod.ext (Fin.ext (by omega))
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

theorem single {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (h : Valid p) (initial current : State α)
    (hout : GridExecution.Outside p.output initial current) (pid : Nat) :
    ∃ final,one f M p pid current=some final ∧
      GridExecution.Post p.output (address p) (fun x => cell f (result f M p initial x))
        (owns p pid) current final := by
  let s := atProgram p pid current
  have outputs (i : Index p) (hi : active p s i) :
      (execution f M p s).val.mem p.output (outputOffset p s i)=cell f (value f M p s i) := by
    rw [execution_memory]
    exact VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_readback_active
      _ _ _ _ _ i hi (local_injective p h s)
  have frame (r o) (outside : r≠p.output ∨ ∀ i,active p s i → o≠outputOffset p s i) :
      (execution f M p s).val.mem r o=s.mem r o := by
    rw [execution_memory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (outside.imp id (fun h i _ => h i)) _
  refine ⟨(execution f M p s).val,(execution f M p s).property,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hr' : row p s i=x.1.val := hr
    have hc' : column p s i=x.2.val := hc
    have hi : active p s i := by simpa [active,hr',hc'] using And.intro x.1.isLt x.2.isLt
    have ho : outputOffset p s i=address p x := by simp [outputOffset,address,hr',hc',h.outputRow,h.outputColumn]
    rw [←ho,outputs i hi]
    congr 1
    simp only [value,result,difference,weightOffset,zeroOffset,scaleOffset,hr',hc']
    cases f <;> simp only [s,atProgram,hout _ _ h.weight,hout _ _ h.zero,hout _ _ h.scale]
  · intro r o outside
    apply frame r o
    rcases outside with hr|ho
    · exact Or.inl hr
    · right
      intro i hi
      let x : Coordinate p := (⟨row p s i,hi.1⟩,⟨column p s i,hi.2⟩)
      have hx : owns p pid x := ⟨i,rfl,rfl⟩
      simpa [x,address,outputOffset,h.outputRow,h.outputColumn] using ho x hx

theorem covers (p : Parameters) (h : Valid p) (x : Coordinate p) :
    ∃ pid∈List.range (count p),owns p pid x := by
  let g := MatmulDequantInt4.rectangularGrid p.rows p.columns p.blockRows p.blockColumns h.br h.bc
  obtain ⟨i,hi⟩ := g.covers ⟨x.1,x.2⟩ ⟨x.1.isLt,x.2.isLt⟩
  refine ⟨i.1.val,List.mem_range.mpr i.1.isLt,(i.2.1,i.2.2,PUnit.unit),?_,?_⟩
  · exact congrArg MatmulDequantInt4.Work.row hi
  · exact congrArg MatmulDequantInt4.Work.col hi

/-- Successful complete DSL launch, every output, and every other typed cell.
No execution-success, output-equality, allocator or host-return premise. -/
theorem whole {α : Type} [Inhabited α] (f : Profile) (M : Algebra α)
    (p : Parameters) (h : Valid p) (initial : State α) :
    ∃ final,run f M p initial=some final ∧
      GridExecution.Post p.output (address p) (fun x => cell f (result f M p initial x))
        (fun _ => True) initial final := by
  obtain ⟨final,executed,values,frame⟩ := GridExecution.compose p.output (address p)
    (address_injective p) (fun x => cell f (result f M p initial x)) (owns p)
    (one f M p) initial (count p)
    (fun pid _ current hout => single f M p h initial current hout pid)
    (List.range (count p)) (fun pid hp => List.mem_range.mp hp) initial (fun _ _ _ => rfl)
  exact ⟨final,executed,fun x _ => values x (covers p h x),
    fun r o ho => frame r o (ho.imp id (fun ho x _ => ho x trivial))⟩

#specNonCircular result avoiding [run,one,source,execution]
#axiomsClean whole
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.KernelGrid
