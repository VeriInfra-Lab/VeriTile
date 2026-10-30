import bench.tritonbench_g.cross_entropy2.IO
import bench.tritonbench_g.cross_entropy2.Proofs.PublicIO
import bench.tritonbench_g.cross_entropy2.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.CrossEntropy2.RealIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000

theorem mapCell_id {α : Type} (cell : Cell α) : mapCell id cell=cell := by
  cases cell with
  | mk d v => cases d <;> rfl

theorem real_cell (cell : Cell (WithBot ℝ)) (raw : WithBot ℝ) (x : ℝ)
    (typed : mapCell id cell=Cell.mk .real raw) (value : cell.read .real=some x) :
    mapCell decode cell=Cell.mk .real x := by
  have eq : cell=Cell.mk .real raw := (mapCell_id cell).symm.trans typed
  rw [eq] at value ⊢
  simp only [Cell.read_same] at value
  rw [value]
  rfl

theorem forward (p : Parameters) (row block : Nat) (hr : row<p.rows)
    (positive : 0<p.block) (start : block*p.block<p.cols) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss) :
    (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some decode).Implements
      (IO.forwardFormula p (fun xs label =>
        (KernelMath.lse p xs block,KernelMath.loss p xs label block,KernelMath.zLoss p xs label block))) := by
  intro vals _ bounds bounded initial binding
  let lifted := forwardValues p vals
  have hb : (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) id id).InBounds bounds lifted := by
    simpa [ProgramIO.InBounds,IO.forward,IO.backward,IO.forwardInputs,IO.backwardInputs,
      lifted,forwardValues,backwardValues] using bounded
  have hi : (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) id id).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  obtain ⟨safe,final,ran,result,frame⟩ := PublicIO.forward KernelReal.algebra p row block
    (KernelReal.arguments p) positive separate outputs lifted trivial bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · simpa [ProgramIO.Frame,ProgramIO.Outside,IO.forward,IO.backward,IO.forwardOutputs] using frame
  let initial' := (IO.launch KernelReal.algebra p (forwardSyntax p) row block (KernelReal.arguments p)).initial initial
  let xs := IO.vector (vals (0 : Fin 2))
  let label := vals (1 : Fin 2) PUnit.unit
  have hx : CrossEntropy1.RealSource.Input p.toParameters (KernelContract.scalarMemory initial') row xs := by
    intro c hc
    simpa [initial',xs,IO.vector,hc,IO.forward,IO.forwardInputs,mapValue,
      CrossEntropy1.Source.logitAddress,KernelContract.scalarMemory,CrossEntropy1.Source.Cell.read] using
      binding (0 : Fin 2) (⟨c,hc⟩,PUnit.unit) trivial
  have ht : CrossEntropy1.Source.label p.toParameters (KernelContract.scalarMemory initial') row=label := by
    simpa [initial',label,CrossEntropy1.Source.label,KernelContract.scalarMemory,
      CrossEntropy1.Source.Cell.read,IO.forward,IO.forwardInputs,mapValue] using
      binding (1 : Fin 2) PUnit.unit trivial
  obtain ⟨other,otherRan,lse,loss,zloss,_⟩ := RawCorrect.forward p separate outputs initial'
    (IOBinding.arguments _ p _ row block _ initial) row block hr rfl rfl xs hx label ht positive start
  have eq : other=final := Option.some.inj (otherRan.symm.trans (by
    simpa only [IO.forward,runSchedule_singleton,Invocation.run,IO.launch,KernelNative.engine] using ran))
  subst other
  intro port lane active
  have typed := result port lane active
  fin_cases port
  all_goals
    change mapCell decode (final.mem _ _)=_
    change mapCell id (final.mem _ _)=_ at typed
    cases lane
  · exact real_cell _ _ _ typed lse
  · exact real_cell _ _ _ typed loss
  · exact real_cell _ _ _ typed zloss

theorem backward (p : Parameters) (row block : Nat) (hr : row<p.rows) :
    (IO.backward KernelReal.algebra p row block (KernelReal.arguments p) some decode).Implements
      (IO.backwardFormula p (fun xs label L upstream lane =>
        KernelMath.gradient p xs label L upstream (block*p.block+lane.val))) := by
  intro vals _ bounds bounded initial binding
  let lifted := backwardValues p vals
  have hb : (IO.backward KernelReal.algebra p row block (KernelReal.arguments p) id id).InBounds bounds lifted := by
    simpa [ProgramIO.InBounds,IO.forward,IO.backward,IO.forwardInputs,IO.backwardInputs,
      lifted,forwardValues,backwardValues] using bounded
  have hi : (IO.backward KernelReal.algebra p row block (KernelReal.arguments p) id id).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  obtain ⟨safe,final,ran,result,frame⟩ := PublicIO.backward KernelReal.algebra p row block
    (KernelReal.arguments p) lifted trivial bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · simpa [ProgramIO.Frame,ProgramIO.Outside,IO.forward,IO.backward,IO.forwardOutputs] using frame
  let initial' := (IO.launch KernelReal.algebra p (backwardSyntax p) row block (KernelReal.arguments p)).initial initial
  let xs := IO.vector (vals (0 : Fin 4))
  let label := vals (1 : Fin 4) PUnit.unit
  let L := vals (2 : Fin 4) PUnit.unit
  let upstream := vals (3 : Fin 4) PUnit.unit
  have hx : CrossEntropy1.RealSource.Input p.toParameters (KernelContract.scalarMemory initial') row xs := by
    intro c hc
    simpa [initial',xs,IO.vector,hc,IO.backward,IO.backwardInputs,mapValue,
      CrossEntropy1.Source.logitAddress,KernelContract.scalarMemory,CrossEntropy1.Source.Cell.read] using
      binding (0 : Fin 4) (⟨c,hc⟩,PUnit.unit) (Or.inl (by simp))
  have ht : CrossEntropy1.Source.label p.toParameters (KernelContract.scalarMemory initial') row=label := by
    simpa [initial',label,CrossEntropy1.Source.label,KernelContract.scalarMemory,
      CrossEntropy1.Source.Cell.read,IO.backward,IO.backwardInputs,mapValue] using
      binding (1 : Fin 4) PUnit.unit (Or.inl (by simp))
  have hl : ((KernelContract.scalarMemory initial') p.lse (p.lseBase+row)).read .real=some L := by
    simpa [initial',L,KernelContract.scalarMemory,CrossEntropy1.Source.Cell.read,
      IO.backward,IO.backwardInputs,mapValue] using
      binding (2 : Fin 4) PUnit.unit (Or.inl (by simp))
  intro port lane active
  fin_cases port
  have typed := result (0 : Fin 1) lane active
  change mapCell decode (final.mem p.gradient
      (p.gradientBase+row*p.gradientStride+(block*p.block+lane.1.val)))=
    .mk .real (KernelMath.gradient p xs label L upstream (block*p.block+lane.1.val))
  by_cases ignored : label=p.ignored
  · change mapCell id (final.mem p.gradient
        (p.gradientBase+row*p.gradientStride+(block*p.block+lane.1.val)))=
        .mk .real (Reference.backward KernelReal.algebra p (KernelReal.arguments p) block
          (IO.vector (lifted (0 : Fin 4))) label (some L) (some upstream) lane.1) at typed
    have zero : Reference.backward KernelReal.algebra p (KernelReal.arguments p) block
        (IO.vector (lifted (0 : Fin 4))) label (some L) (some upstream) lane.1=some (0:ℝ) := by
      simp only [Reference.backward,ignored,if_pos,KernelReal.algebra,CrossEntropy1.KernelReal.algebra]
      norm_num
      rfl
    rw [zero] at typed
    have readZero : (final.mem p.gradient
        (p.gradientBase+row*p.gradientStride+(block*p.block+lane.1.val))).read .real=some (0:ℝ) := by
      have reads := congrArg (fun c : Cell (WithBot ℝ) => c.read .real) typed
      simpa only [mapCell_id,Cell.read_same] using reads
    simpa only [KernelMath.gradient,Math.gradientWithLSE,ignored,if_pos] using
      real_cell _ _ _ typed readZero
  · have hu : ((KernelContract.scalarMemory initial') p.upstream
        (p.upstreamBase+row*p.upstreamStride)).read .real=some upstream := by
      simpa [initial',upstream,KernelContract.scalarMemory,CrossEntropy1.Source.Cell.read,
        IO.backward,IO.backwardInputs,mapValue] using
        binding (3 : Fin 4) PUnit.unit (Or.inr ignored)
    obtain ⟨other,otherRan,values,_⟩ := RawCorrect.backward p initial'
      (IOBinding.arguments _ p _ row block _ initial) row block hr rfl rfl xs L upstream hx label ht hl hu
    have eq : other=final := Option.some.inj (otherRan.symm.trans (by
      simpa only [IO.backward,runSchedule_singleton,Invocation.run,IO.launch,KernelNative.engine] using ran))
    subst other
    exact real_cell _ _ _ typed (values lane.1 active)

#axiomsClean forward
#axiomsClean backward

end
end VeriTile.Bench.Repairs.CrossEntropy2.RealIO
