import bench.tritonbench_g.cross_entropy1.Proofs.PublicIO
import bench.tritonbench_g.cross_entropy1.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000

def decode (x : WithBot ℝ) : ℝ := x.unbotD 0

theorem mapCell_id {α : Type} (cell : Cell α) : mapCell id cell=cell := by
  cases cell with | mk d v => cases d <;> rfl

theorem real_cell (cell : Cell (WithBot ℝ)) (raw : WithBot ℝ) (x : ℝ)
    (typed : mapCell id cell=Cell.mk .real raw) (value : cell.read .real=some x) :
    mapCell decode cell=Cell.mk .real x := by
  have eq : cell=Cell.mk .real raw := (mapCell_id cell).symm.trans typed
  rw [eq] at value ⊢
  simp only [Cell.read_same] at value
  rw [value]
  rfl

def forwardValues (p : Parameters) (xs : ChannelValues ℝ (IO.forwardInputs p)) :
    ChannelValues (WithBot ℝ) (IO.forwardInputs p) :=
  Fin.cases (fun j => some (xs (0 : Fin 2) j))
    (fun next => Fin.cases (fun j => xs (1 : Fin 2) j) (fun impossible => Fin.elim0 impossible) next)
def backwardValues (p : Parameters) (xs : ChannelValues ℝ (IO.backwardInputs p)) :
    ChannelValues (WithBot ℝ) (IO.backwardInputs p) :=
  Fin.cases (fun j => some (xs (0 : Fin 4) j))
    (fun next => Fin.cases (fun j => xs (1 : Fin 4) j)
      (fun next => Fin.cases (fun j => some (xs (2 : Fin 4) j))
        (fun next => Fin.cases (fun j => some (xs (3 : Fin 4) j))
          (fun impossible => Fin.elim0 impossible) next) next) next)

def unsplitDomain (p : Parameters) (v : ChannelValues ℝ (IO.forwardInputs p)) : Prop :=
  let label : Int := v (1 : Fin 2) PUnit.unit
  label=p.ignored ∨ (0≤label ∧ label<(p.cols : Int))

theorem forward_of_reads (p : Parameters) (row block : Nat) (positive : 0<p.block)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (domain : ChannelValues ℝ (IO.forwardInputs p) → Prop)
    (f : (Nat → ℝ) → Int → ℝ × ℝ)
    (correct : ∀ vals,domain vals → ∀ initial : State (WithBot ℝ),
      RuntimeArgs.Bound p initial (KernelReal.arguments p) →
      initial.pids 0=row → initial.pids 1=block →
      RealSource.Input p (KernelContract.scalarMemory initial) row (IO.vector (vals (0 : Fin 2))) →
      Source.label p (KernelContract.scalarMemory initial) row=vals (1 : Fin 2) PUnit.unit →
      ∃ final,exec (KernelNative.engine KernelReal.algebra) (forwardSyntax p) initial=some final ∧
        (final.mem p.lse (p.lseBase+block*p.rows+row)).read .real=
          some (f (IO.vector (vals (0 : Fin 2))) (vals (1 : Fin 2) PUnit.unit)).1 ∧
        (final.mem p.loss (p.lossBase+block*p.rows+row)).read .real=
          some (f (IO.vector (vals (0 : Fin 2))) (vals (1 : Fin 2) PUnit.unit)).2) :
    (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some decode).ImplementsOn domain
      (IO.forwardFormula p f) := by
  intro vals hd bounds bounded initial binding
  let lifted := forwardValues p vals
  have hb : (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) id id).InBounds bounds lifted := by
    simpa [ProgramIO.InBounds,IO.forward,IO.forwardInputs,lifted,forwardValues] using bounded
  have hi : (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) id id).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  obtain ⟨safe,final,ran,result,frame⟩ := PublicIO.forward KernelReal.algebra p row block
    (KernelReal.arguments p) positive separate outputs lifted trivial bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · simpa [ProgramIO.Frame,ProgramIO.Outside,IO.forward,IO.forwardOutputs] using frame
  let initial' := (IO.launch KernelReal.algebra p (forwardSyntax p) row block (KernelReal.arguments p)).initial initial
  have hx : RealSource.Input p (KernelContract.scalarMemory initial') row (IO.vector (vals (0 : Fin 2))) := by
    intro c hc
    simpa [initial',IO.vector,hc,IO.forward,IO.forwardInputs,mapValue,
      Source.logitAddress,KernelContract.scalarMemory,Source.Cell.read] using
      binding (0 : Fin 2) (⟨c,hc⟩,PUnit.unit) trivial
  have ht : Source.label p (KernelContract.scalarMemory initial') row=vals (1 : Fin 2) PUnit.unit := by
    simpa [initial',Source.label,KernelContract.scalarMemory,Source.Cell.read,IO.forward,IO.forwardInputs,mapValue] using
      binding (1 : Fin 2) PUnit.unit trivial
  obtain ⟨other,otherRan,lse,loss⟩ := correct vals hd initial'
    (IOBinding.arguments _ p _ row block _ initial) rfl rfl hx ht
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

theorem forward_unsplit (p : Parameters) (row : Nat) (hr : row<p.rows)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols) :
    (IO.forward KernelReal.algebra p row 0 (KernelReal.arguments p) some decode).ImplementsOn
      (unsplitDomain p) (IO.forwardFormula p (fun xs label =>
        (Math.lse p.cols xs,Math.loss p.cols xs label p.ignored p.smoothing p.squareScale))) := by
  apply forward_of_reads p row 0 (by omega) separate outputs
  intro vals hd initial abi rowPID blockPID input label
  have valid : Source.label p (KernelContract.scalarMemory initial) row=p.ignored ∨
      (0≤Source.label p (KernelContract.scalarMemory initial) row ∧
        Source.label p (KernelContract.scalarMemory initial) row<(p.cols : Int)) := by
    simpa only [label,unsplitDomain] using hd
  obtain ⟨final,ran,lse,loss,_⟩ := RawCorrect.forward_unsplit_kernel p separate outputs initial abi row hr
    rowPID blockPID _ input positive covered splitFlag noSmooth startZero total valid
  exact ⟨final,ran,by simpa only [Nat.zero_mul,Nat.add_zero] using lse,
    by simpa only [Nat.zero_mul,Nat.add_zero,label] using loss⟩

theorem forward_split (p : Parameters) (row block : Nat) (hr : row<p.rows)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (positive : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    (IO.forward KernelReal.algebra p row block (KernelReal.arguments p) some decode).Implements
      (IO.forwardFormula p (fun xs label =>
        (Math.blockLSE p.cols p.block block xs,RealSource.partialLoss p xs label block))) := by
  apply forward_of_reads p row block positive separate outputs (fun _ => True)
  intro vals _ initial abi rowPID blockPID input label
  obtain ⟨final,ran,lse,loss,_⟩ := RawCorrect.forward_split_kernel p separate outputs initial abi row block hr
    rowPID blockPID _ input positive start splitFlag noSmooth
  exact ⟨final,ran,lse,by simpa only [label] using loss⟩

theorem backward (p : Parameters) (row block : Nat)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    (IO.backward KernelReal.algebra p row block (KernelReal.arguments p) some decode).Implements
      (IO.backwardFormula p (fun xs label L upstream lane =>
        KernelMath.gradient p xs label L upstream (block*p.block+lane.val))) := by
  intro vals _ bounds bounded initial binding
  let lifted := backwardValues p vals
  have hb : (IO.backward KernelReal.algebra p row block (KernelReal.arguments p) id id).InBounds bounds lifted := by
    simpa [ProgramIO.InBounds,IO.backward,IO.backwardInputs,lifted,backwardValues] using bounded
  have hi : (IO.backward KernelReal.algebra p row block (KernelReal.arguments p) id id).InputBinding lifted initial := by
    intro port lane active
    fin_cases port <;> exact binding _ lane active
  obtain ⟨safe,final,ran,result,frame⟩ := PublicIO.backward KernelReal.algebra p row block
    (KernelReal.arguments p) lifted trivial bounds hb initial hi
  refine ⟨safe,final,ran,?_,?_⟩
  swap
  · simpa [ProgramIO.Frame,ProgramIO.Outside,IO.backward] using frame
  let initial' := (IO.launch KernelReal.algebra p (backwardSyntax p) row block (KernelReal.arguments p)).initial initial
  let xs := IO.vector (vals (0 : Fin 4))
  let label := vals (1 : Fin 4) PUnit.unit
  let L := vals (2 : Fin 4) PUnit.unit
  let upstream := vals (3 : Fin 4) PUnit.unit
  have hx : RealSource.Input p (KernelContract.scalarMemory initial') row xs := by
    intro c hc
    simpa [initial',xs,IO.vector,hc,IO.backward,IO.backwardInputs,mapValue,
      Source.logitAddress,KernelContract.scalarMemory,Source.Cell.read] using
      binding (0 : Fin 4) (⟨c,hc⟩,PUnit.unit) (Or.inl (by simp))
  have ht : Source.label p (KernelContract.scalarMemory initial') row=label := by
    simpa [initial',label,Source.label,KernelContract.scalarMemory,
      Source.Cell.read,IO.backward,IO.backwardInputs,mapValue] using
      binding (1 : Fin 4) PUnit.unit (Or.inl (by simp))
  have hl : ((KernelContract.scalarMemory initial') p.lse (p.lseBase+row)).read .real=some L := by
    simpa [initial',L,KernelContract.scalarMemory,Source.Cell.read,
      IO.backward,IO.backwardInputs,mapValue] using
      binding (2 : Fin 4) PUnit.unit (Or.inl (by simp))
  have hu : Source.label p (KernelContract.scalarMemory initial') row≠p.ignored →
      ((KernelContract.scalarMemory initial') p.upstream
        (p.upstreamBase+row*p.upstreamStride)).read .real=some upstream := by
    intro active
    simpa [initial',upstream,KernelContract.scalarMemory,Source.Cell.read,
      IO.backward,IO.backwardInputs,mapValue] using
      binding (3 : Fin 4) PUnit.unit (Or.inr (by simpa only [ht] using active))
  obtain ⟨other,otherRan,values,_⟩ := RawCorrect.backward_kernel p initial'
    (IOBinding.arguments _ p _ row block _ initial) row block rfl rfl xs L upstream hx hl hu noSmooth
  have eq : other=final := Option.some.inj (otherRan.symm.trans (by
    simpa only [IO.backward,runSchedule_singleton,Invocation.run,IO.launch,KernelNative.engine] using ran))
  subst other
  intro port lane active
  fin_cases port
  have typed := result (0 : Fin 1) lane active
  have computed := values lane.1 active
  rw [ht] at computed
  exact real_cell _ _ _ typed computed

#axiomsClean forward_unsplit
#axiomsClean forward_split
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealIO
