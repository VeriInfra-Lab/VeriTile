import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.TensorChecks
import bench.repairs.tritonbench_g.attention_fwd_triton1_tile_checked.Proofs.Projection

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Physical
open VeriTile.Triton FP.Structural
open AttentionFwdTriton1.Whole001
noncomputable section
set_option maxHeartbeats 8000000

/-- A physical FP32 cell is addressed by its byte start. Other cells, including
non-word starts and cells of any dtype, are retained by the frame theorem. -/
def WordAt (a : Allocation) (address : Nat) : Prop :=
  address%4=0 ∧ a.Contains (address/4)
instance (a : Allocation) (address : Nat) : Decidable (WordAt a address) :=
  inferInstanceAs (Decidable (address%4=0 ∧ a.base≤address/4 ∧ address/4<a.base+a.length))

def project {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (storeFlag : Bool) (before : Cells A)
    (raw : State (Term A)) : Cells A := fun address =>
  if Host.Empty c then before address
  else if WordAt abi.o address then
    .mk .fp32 (((raw.mem "o" (address/4-abi.o.base)).read .real).evaluate ops literal)
  else if storeFlag=Bool.true ∧ WordAt abi.h address then
    .mk .fp32 (((raw.mem "h" (address/4-abi.h.base)).read .real).evaluate ops literal)
  else before address

def rawResult {A} [Inhabited A] (c : Shape) (abi : ABI c) (scale : ℝ)
    (storeFlag ifcond : Bool) (before : Cells A) (rest : State (Term A)) : State (Term A) :=
  SharedGrid.result c scale storeFlag ifcond (initial c abi before rest)
def result {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) : Cells A :=
  project ops literal c abi storeFlag before (rawResult c abi scale storeFlag ifcond before rest)

/-- Executes the actual full quote in the shared interpreter, then realizes its
logical-region writes at the observed ABI byte addresses. -/
def run {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) : Option (Cells A) := do
  let raw ← SharedGrid.run c scale storeFlag ifcond (initial c abi before rest)
  pure (project ops literal c abi storeFlag before raw)

theorem success {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) :
    run ops literal c abi scale storeFlag ifcond before rest=
      some (result ops literal c abi scale storeFlag ifcond before rest) := by
  simp only [run,SharedGrid.run_success]
  rfl

theorem wordAt (a : Allocation) (i : Nat) (hi : i<a.length) : WordAt a (4*(a.base+i)) := by
  simp [WordAt,Allocation.Contains,hi]

theorem separate (a b : Allocation) (sep : a.Separate b) (address : Nat)
    (ha : WordAt a address) : ¬WordAt b address := fun hb => sep _ ha.2 hb.2

theorem output_cell {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (storeFlag : Bool) (before : Cells A) (raw : State (Term A))
    (live : ¬Host.Empty c) (i : Nat) (hi : i<c.mainSize) :
    project ops literal c abi storeFlag before raw (4*(abi.o.base+i))=
      .mk .fp32 (((raw.mem "o" i).read .real).evaluate ops literal) := by
  have atWord := wordAt abi.o i (by rwa [abi.oLength])
  simp [project,live,atWord]

theorem history_cell {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (before : Cells A) (raw : State (Term A))
    (live : ¬Host.Empty c) (i : Nat) (hi : i<c.historySize) :
    project ops literal c abi Bool.true before raw (4*(abi.h.base+i))=
      .mk .fp32 (((raw.mem "h" i).read .real).evaluate ops literal) := by
  have atWord := wordAt abi.h i (by rwa [abi.hLength])
  have outside := separate abi.h abi.o (abi.historySeparate abi.o (by simp)) _ atWord
  simp [project,live,atWord,outside]

theorem frame {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) (address : Nat)
    (ho : ¬WordAt abi.o address) (hh : storeFlag=Bool.false ∨ ¬WordAt abi.h address) :
    result ops literal c abi scale storeFlag ifcond before rest address=before address := by
  unfold result project
  rcases hh with hh|hh <;> simp [ho,hh]

theorem empty {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) (he : Host.Empty c) :
    result ops literal c abi scale storeFlag ifcond before rest=before := by
  funext address
  simp [result,project,he]

theorem inputs {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A)
    (c : Shape) (abi : ABI c) (before : Cells A) (rest : State (Term A)) :
    Projection.inputs ops literal c
      (SharedStep.read (initial c abi before rest) "q")
      (SharedStep.read (initial c abi before rest) "k")
      (SharedStep.read (initial c abi before rest) "v")=tensorInputs c abi before := by
  rfl

theorem output {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A) (zero : literal 0=ops.zero)
    (c : Shape) (abi : ABI c) (scale : ℝ) (storeFlag ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) (head row : Nat) (j : Fin c.D)
    (hh : head<c.heads) (hr : row<c.T) :
    result ops literal c abi scale storeFlag ifcond before rest (4*(abi.o.base+c.mainOffset head row j.val))=
      .mk .fp32 (ops.cast (Reference.output ops c (tensorInputs c abi before)
        (literal scale) head (row/32) ifcond (row%32) j.val)) := by
  have live : ¬Host.Empty c := by unfold Host.Empty; omega
  rw [result,output_cell _ _ _ _ _ _ _ live _ (c.main_bound _ _ _ hh hr j.isLt)]
  unfold rawResult
  rw [SharedGrid.result,if_neg (show ¬SharedGrid.Empty c from live),
    SharedGrid.output c scale storeFlag ifcond _ c.heads head hh row hr j]
  change Cell.mk .fp32 (ops.cast ((Graph.output c _ _ _ scale head (row/32) ifcond _).evaluate ops literal))=_
  rw [Projection.graph_output ops literal zero,inputs]

theorem history {A} [Inhabited A] (ops : Ops A) (literal : ℝ → A) (zero : literal 0=ops.zero)
    (c : Shape) (abi : ABI c) (scale : ℝ) (ifcond : Bool)
    (before : Cells A) (rest : State (Term A)) (head chunk : Nat) (i j : Fin c.D)
    (hh : head<c.heads) (hc : chunk<c.chunks) :
    result ops literal c abi scale Bool.true ifcond before rest
      (4*(abi.h.base+c.historyOffset head chunk i.val j.val))=
      .mk .fp32 (ops.cast (Reference.carry ops c (tensorInputs c abi before)
        (literal scale) head ifcond chunk i.val j.val)) := by
  have ht : c.T≠0 := by intro h; have := (c.chunks_zero).2 h; omega
  have live : ¬Host.Empty c := by unfold Host.Empty; omega
  rw [result,history_cell _ _ _ _ _ _ live _ (c.history_bound _ _ _ _ hh hc i.isLt j.isLt)]
  unfold rawResult
  rw [SharedGrid.result,if_neg (show ¬SharedGrid.Empty c from live),
    SharedGrid.history c scale ifcond _ c.heads head hh chunk hc i j]
  change Cell.mk .fp32 (ops.cast ((Graph.carry c _ _ head ifcond chunk _).evaluate ops literal))=_
  rw [Projection.graph_carry ops literal zero c
    (SharedStep.read (initial c abi before rest) "q") _ _ scale,inputs]

#axiomsClean success
#axiomsClean output
#axiomsClean history
#axiomsClean frame
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1TileChecked.Physical
