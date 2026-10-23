import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.Grid
import bench.repairs.tritonbench_g.matmul_triton1.revision001.Proofs.FiniteExecution
namespace VeriTile.Bench.Repairs.MatmulTriton2.Revision001.FiniteSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.Repairs.MatmulTriton1.Revision001
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Canonical (p : Core.Parameters) : Prop where
  inputRow : p.inputRow=p.innerSize
  inputInner : p.inputInner=1
  weightInner : p.weightInner=p.columns
  weightColumn : p.weightColumn=1
  outputRow : p.outputRow=p.columns
  outputColumn : p.outputColumn=1

theorem a_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) (k : Nat) :
    FiniteExecution.naturalA P p Bool.true s k = Core.aValues P p s k := by
  funext i
  unfold FiniteExecution.naturalA Core.aValues
  change (if Core.row p s i.1 < p.rows ∧ i.2.1.val < Core.inner p-k*p.blockInner then
    (s.mem p.input (Core.row p s i.1*p.innerSize+(k*p.blockInner+i.2.1.val))).read .real else _) = _
  simp only [Core.aOffset,h.inputRow,h.inputInner,Nat.mul_one,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem b_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) (k : Nat) :
    FiniteExecution.naturalB P p Bool.true s k = Core.bValues P p s k := by
  funext i
  unfold FiniteExecution.naturalB Core.bValues
  change (if i.1.val < Core.inner p-k*p.blockInner ∧ Core.column p s i.2.1 < p.columns then
    (s.mem p.weight ((k*p.blockInner+i.1.val)*p.columns+Core.column p s i.2.1)).read .real else _) = _
  simp only [Core.bOffset,h.weightInner,h.weightColumn,Nat.mul_one,Nat.add_mul,
    Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem accumulation_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) (k : Nat) :
    FiniteExecution.accumulation P p Bool.true s k = Core.accumulation P p s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [FiniteExecution.accumulation,Core.accumulation,a_eq P p h,b_eq P p h,ih]
    simp [Core.matrixDot,MatmulTriton1.Revision001.Core.matrixDot]

theorem written_eq (P : Core.Primitives α) (p : Core.Parameters) (h : Canonical p)
    (s : State α) : FiniteExecution.written P p Bool.true s = Core.written P p s := by
  unfold FiniteExecution.written Core.written
  congr 1
  funext t i
  have active : FiniteExecution.active p Bool.true s i = Core.active p s i := rfl
  have offset : FiniteExecution.offset p Bool.true s i = Core.cOffset p s i := by
    change Core.row p s i.1*p.columns+Core.column p s i.2.1 = _
    simp [Core.cOffset,h.outputRow,h.outputColumn]
  simp only [active,offset,accumulation_eq P p h]
  rfl

/-- Complete typed-source execution has the same memory as its proved
load/dot/store summary. This exposes the interpreter, rather than assuming a
summary-to-source correspondence. -/
theorem source_memory (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (s : State α) (hk : 0 < p.blockInner) :
    ∃ final,FP.Structural.exec (Core.backend rest P) (Core.source p) s=some final ∧
      final.mem=(Core.written P p s).mem := by
  let start := Core.preludeExecution rest P p s
  obtain ⟨middle,hm,hi⟩ := Core.loopExecution rest P p s start.val 0 (by omega)
    (Core.preludeInvariant rest P p s)
  let finish := Core.epilogueExecution rest P p s middle hi
  refine ⟨finish.val,?_,Core.epilogueMemory rest P p s middle hi⟩
  change run (Core.backend rest P) (Core.source p).surfaceBody s=some finish.val
  rw [Core.source_decomposition,run_append,run_append,start.property]
  have hs (t : State α) : run (Core.backend rest P) [Core.loopStmt p] t =
      FP.Structural.range (Core.backend rest P) "k" 0 (Core.steps p) 1 (Core.body p) t := by
    simp [run,Core.dynamicLoop rest P p t hk]
  simp only [Option.bind_some,hs,hm]
  exact finish.property

/-- The finite64 executor and the complete typed source agree on every cell.
The typed source keeps its own internal registers; only memory is observable. -/
theorem program_source (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : Canonical p) (h : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (s : State α) (hp : s.pids 0 < FiniteIndex.count p) :
    ∃ actual typed,
      FiniteExecution.program P p Bool.true s=some actual ∧
      FP.Structural.exec (Core.backend rest P) (Core.source p) s=some typed ∧ actual.mem=typed.mem := by
  obtain ⟨typed,ht,hm⟩ := source_memory rest P p s (by have := h.blockInner.1; omega)
  refine ⟨Core.written P p s,typed,?_,ht,hm.symm⟩
  rw [FiniteExecution.program_eq P p h storage Bool.true s hp,written_eq P p canonical]

def one (P : Core.Primitives α) (p : Core.Parameters) (pid : Nat) (s : State α) :=
  FiniteExecution.program P p Bool.true (GridExecution.atProgram s pid)
def grid (P : Core.Primitives α) (p : Core.Parameters) (s : State α) :=
  GridExecution.sequence (one P p) (List.range (Geometry.count (Core.geometry p))) s

theorem single_logical (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : Canonical p) (h : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (hgeo : Geometry.Valid (Core.geometry p)) (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial current : State α) (outside : GridExecution.Outside p.output initial current)
    (pid : Nat) (hp : pid < Geometry.count (Core.geometry p)) :
    ∃ final,one P p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (Core.geometry p))
        (fun x : Core.LogicalIndex p => Cell.mk .real (Core.logical P p initial x.1.val x.2.val))
        (Geometry.Owns (Core.geometry p) pid) current final := by
  obtain ⟨actual,typed,he,ht,hm⟩ := program_source rest P p canonical h storage
    (GridExecution.atProgram current pid) hp
  obtain ⟨other,ho,post⟩ := Core.single_logical rest P p hgeo
    (by have := h.blockInner.1; omega) ha hb initial current outside pid
  have same : typed=other := Option.some.inj (ht.symm.trans ho)
  subst other
  refine ⟨actual,he,?_⟩
  simpa only [GridExecution.Post,hm] using post

theorem grid_correct (rest : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (canonical : Canonical p) (h : FiniteIndex.Ready p) (storage : FiniteIndex.StorageBounds p)
    (hgeo : Geometry.Valid (Core.geometry p)) (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (initial : State α) :
    ∃ final,grid P p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (Core.geometry p))
        (fun x : Core.LogicalIndex p => Cell.mk .real (Core.logical P p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hr,hf⟩ := GridExecution.compose p.output (Geometry.address (Core.geometry p))
    hgeo.writable (fun x : Core.LogicalIndex p => Cell.mk .real (Core.logical P p initial x.1.val x.2.val))
    (Geometry.Owns (Core.geometry p)) (one P p) initial (Geometry.count (Core.geometry p))
    (fun pid hp current hout => single_logical rest P p canonical h storage hgeo ha hb initial current hout pid hp)
    (List.range (Geometry.count (Core.geometry p))) (fun _ hp => List.mem_range.mp hp)
    initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (Core.geometry p) hgeo x
    exact hr x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o outside
    exact hf r o (outside.imp id (fun hx x _ => hx x trivial))

#axiomsClean program_source
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.MatmulTriton2.Revision001.FiniteSource
