import bench.repairs.tritonbench_g.kldiv_compute.Proofs.KernelBinding
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Program

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open Contiguous
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Omitted tl.load padding is arbitrary at every dtype, region and offset. -/
abbrev Undefined (α : Type) := (dtype : TileDType) → RegionName → Nat → Value α dtype

abbrev parameters (n : Nat) : Parameters := ⟨"x","y","out",n,1024⟩

/-- Bind all source regions to one physical memory, including aliases of the
read-only inputs. Undefined inactive load values remain uninterpreted. -/
def physicalState {M α β : Type} [Inhabited α] (ops : Ops M α β)
    (undefined : Undefined α) (memory : M) (x y output : Tensor) (pid : Nat) : State α where
  mem := fun r i => .mk .real (ops.read memory
    (directKey (if r="x" then x else if r="y" then y else output) i))
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then pid else 0
  numPids := fun ax => if ax=0 then gridCount x.metadata.numel else 1
  undef := undefined

def point {M α β : Type} (ops : Ops M α β) (a : Algebra α)
    (memory : M) (x y : Tensor) (i : Nat) : α :=
  Program.scalar a (ops.read memory (directKey x i)) (ops.read memory (directKey y i))

/-- Execute every typed instruction; commit only the source's active stores.
The next program is evaluated against that resulting physical memory. -/
def runProgram {M α β : Type} [Inhabited α] (ops : Ops M α β) (a : Algebra α)
    (undefined : Undefined α) (x y output : Tensor) (pid : Nat) (memory : M) : Option M := do
  let final ← FP.Structural.exec a (KernelDSL.kernel (parameters x.metadata.numel))
    (physicalState ops undefined memory x y output pid)
  pure (stores ops (directKey output) (fun i => (final.mem "out" i).read .real)
    (Index.indices x.metadata.numel pid) memory)

def runJobs {M α β : Type} [Inhabited α] (ops : Ops M α β) (a : Algebra α)
    (undefined : Undefined α) (x y output : Tensor) : List Nat → M → Option M
  | [],memory => some memory
  | pid::rest,memory => do
      let next ← runProgram ops a undefined x y output pid memory
      runJobs ops a undefined x y output rest next

theorem stores_congr {M α β ι : Type} (ops : Ops M α β) (keys : ι → Key)
    (xs ys : ι → α) (jobs : List ι) (memory : M)
    (same : ∀ i ∈ jobs, xs i = ys i) :
    stores ops keys xs jobs memory = stores ops keys ys jobs memory := by
  induction jobs generalizing memory with
  | nil => rfl
  | cons i rest ih =>
    simp only [stores]
    rw [same i (by simp)]
    exact ih _ (fun j hj => same j (by simp [hj]))

theorem program_complete {M α β : Type} [Inhabited α] (ops : Ops M α β) (a : Algebra α)
    (undefined : Undefined α) (x y output : Tensor) (pid : Nat) (valid : Index.ValidJob x.metadata.numel pid) (memory : M) :
    runProgram ops a undefined x y output pid memory =
      some (stores ops (directKey output) (point ops a memory x y)
        (Index.indices x.metadata.numel pid) memory) := by
  obtain ⟨final,executed,values⟩ := Program.complete_correct a (parameters x.metadata.numel)
    (physicalState ops undefined memory x y output pid)
  unfold runProgram
  rw [KernelDSL.source_eq,executed]
  simp only [bind,Option.bind,pure]
  congr 1
  apply stores_congr
  intro i hi
  obtain ⟨lane,hl,rfl⟩ := List.mem_map.mp hi
  obtain ⟨laneBound,active⟩ := (Index.active_iff _ _ valid lane).mp hl
  rw [values,Program.expected_cell]
  simp only [parameters,physicalState,if_true]
  rw [if_pos (by exact ⟨trivial,by omega,by omega,active⟩)]
  simp [Program.resultCell,point,Cell.read]

/-- Complete physical complement, including every raw read and observed byte
outside the active destination. Real observations need no finite byte codec. -/
def Frame {M α β : Type} (ops : Ops M α β) (output : Tensor) (n : Nat)
    (before after : M) : Prop :=
  (∀ observed, (∀ i, i<n → observed.disjoint (directKey output i)) →
    ops.read after observed = ops.read before observed) ∧
  (∀ owner address, (∀ i, i<n → ¬(directKey output i).covers owner address) →
    ops.observe after owner address = ops.observe before owner address)

theorem Frame.refl {M α β : Type} (ops : Ops M α β) (output : Tensor) (n : Nat) (s : M) :
    Frame ops output n s s := ⟨fun _ _ => rfl,fun _ _ _ => rfl⟩

theorem Frame.trans {M α β : Type} {ops : Ops M α β} {output : Tensor} {n : Nat} {a b c : M}
    (ab : Frame ops output n a b) (bc : Frame ops output n b c) : Frame ops output n a c :=
  ⟨fun k outside => (bc.1 k outside).trans (ab.1 k outside),
    fun owner address outside => (bc.2 owner address outside).trans (ab.2 owner address outside)⟩

theorem stores_frame {M α β : Type} (ops : Ops M α β) (output : Tensor) (n : Nat)
    (values : Nat → α) (jobs : List Nat) (memory : M) (valid : ∀ i∈jobs,i<n) :
    Frame ops output n memory (stores ops (directKey output) values jobs memory) := by
  constructor
  · intro observed outside
    exact read_frame ops _ values jobs memory observed (fun i hi => outside i (valid i hi))
  · intro owner address outside
    exact bytes_frame ops _ values jobs memory owner address (fun i hi => outside i (valid i hi))

theorem point_preserved {M α β : Type} (ops : Ops M α β) (a : Algebra α)
    (x y output : Tensor) (domain : Domain x y output) (before after : M)
    (frame : Frame ops output x.metadata.numel before after) (i : Nat) (hi : i<x.metadata.numel) :
    point ops a after x y i = point ops a before x y i := by
  unfold point
  have readX := frame.1 (directKey x i) (fun j hj =>
    (allocation x).spans_separate (allocation output) domain.freshX _ _
      (direct_backed x domain.xLaws domain.admitted.contiguous.1 i hi)
      (direct_backed output domain.outputLaws domain.outputContiguous j (by rw [output_count domain];exact hj)))
  have readY := frame.1 (directKey y i) (fun j hj =>
    (allocation y).spans_separate (allocation output) domain.freshY _ _
      (direct_backed y domain.yLaws domain.admitted.contiguous.2 i
        (by rw [←same_numel x.metadata y.metadata domain.admitted];exact hi))
      (direct_backed output domain.outputLaws domain.outputContiguous j (by rw [output_count domain];exact hj)))
  rw [readX,readY]

theorem stores_read {M α β : Type} (ops : Ops M α β) (output : Tensor)
    (values : Nat → α) (jobs : List Nat) (memory : M) (i : Nat) :
    ops.read (stores ops (directKey output) values jobs memory) (directKey output i) =
      if i∈jobs then values i else ops.read memory (directKey output i) := by
  induction jobs generalizing memory with
  | nil => simp [stores]
  | cons j rest ih =>
    rw [stores,ih]
    by_cases hr : i∈rest
    · simp [hr]
    · by_cases same : i=j
      · subst j
        simp [hr,ops.read_same]
      · simp only [List.mem_cons,hr,same,or_false,if_false]
        exact ops.read_other _ _ _ _ (direct_disjoint output i j same)

theorem jobs_complete {M α β : Type} [Inhabited α] (ops : Ops M α β) (a : Algebra α)
    (undefined : Undefined α) (x y output : Tensor) (domain : Domain x y output) (jobs : List Nat)
    (valid : ∀ pid∈jobs, Index.ValidJob x.metadata.numel pid) (memory : M) :
    ∃ final, runJobs ops a undefined x y output jobs memory = some final ∧
      (∀ i,i<x.metadata.numel → ops.read final (directKey output i) =
        if ∃ pid∈jobs,i∈Index.indices x.metadata.numel pid then point ops a memory x y i
        else ops.read memory (directKey output i)) ∧ Frame ops output x.metadata.numel memory final := by
  induction jobs generalizing memory with
  | nil => exact ⟨memory,rfl,by simp,Frame.refl _ _ _ _⟩
  | cons pid rest ih =>
    let mid := stores ops (directKey output) (point ops a memory x y) (Index.indices x.metadata.numel pid) memory
    have executed := program_complete ops a undefined x y output pid (valid pid (by simp)) memory
    have firstFrame := stores_frame ops output x.metadata.numel (point ops a memory x y)
      (Index.indices x.metadata.numel pid) memory (Index.indices_bound _ _ (valid pid (by simp)))
    obtain ⟨final,lastExec,values,lastFrame⟩ := ih (fun j hj => valid j (by simp [hj])) mid
    refine ⟨final,?_,?_,firstFrame.trans lastFrame⟩
    · simp only [runJobs,executed,bind,Option.bind]
      exact lastExec
    · intro i hi
      rw [values i hi,point_preserved ops a x y output domain memory mid firstFrame i hi]
      have firstValue := stores_read ops output (point ops a memory x y) (Index.indices x.metadata.numel pid) memory i
      change ops.read mid (directKey output i) = _ at firstValue
      rw [firstValue]
      by_cases later : ∃ j∈rest,i∈Index.indices x.metadata.numel j
      · simp [later]
      · by_cases here : i∈Index.indices x.metadata.numel pid <;> simp [later,here]

structure Post {M α β : Type} [Inhabited α] (ops : Ops M α β) (negate : α → α)
    (a : Algebra α) (undefined : Undefined α) (x y output : Tensor) (schedule : Index.Schedule x.metadata.numel)
    (expected : Nat → α) (before after : M) : Prop where
  executes : runJobs ops a undefined x y output schedule.programs before = some after
  values : ∀ i,i<x.metadata.numel → logical ops negate after output i = expected i
  frame : Frame ops output x.metadata.numel before after
  xAddresses : Addresses x
  yAddresses : Addresses y
  outputAddresses : Addresses output
  control : Control x.metadata.numel
  sourcePointers : SourcePointers x y output

/-- Complete launch in any permutation of the requested programs. The actual
source succeeds internally; no output/execution/frame law is a premise. -/
theorem complete {M α β : Type} [Inhabited α] (ops : Ops M α β) (negate : α → α)
    (a : Algebra α) (undefined : Undefined α) (x y output : Tensor) (domain : Domain x y output)
    (schedule : Index.Schedule x.metadata.numel) (memory : M) (expected : Nat → α)
    (formula : ∀ i,i<x.metadata.numel → Program.scalar a (logical ops negate memory x i)
      (logical ops negate memory y i) = expected i) :
    ∃ final, Post ops negate a undefined x y output schedule expected memory final := by
  obtain ⟨final,executed,values,frame⟩ := jobs_complete ops a undefined x y output domain
    schedule.programs (Index.valid_of_mem _ schedule) memory
  have xa := addresses x domain.xLaws domain.admitted.contiguous.1 (fun h => (domain.admitted.alignment h).1)
  have ya : Addresses y := addresses y domain.yLaws domain.admitted.contiguous.2 (fun positive =>
    (domain.admitted.alignment (by rw [same_numel x.metadata y.metadata domain.admitted];exact positive)).2)
  have oa := addresses output domain.outputLaws domain.outputContiguous domain.outputAligned
  refine ⟨final,executed,?_,frame,xa,ya,oa,control domain,source_pointers domain xa ya oa⟩
  · intro i hi
    rw [←direct_logical ops negate final output domain.outputLaws domain.outputContiguous
      domain.outputNoNeg i (by rw [output_count domain];exact hi)]
    have covered : ∃ pid∈schedule.programs,i∈Index.indices x.metadata.numel pid :=
      Index.covered _ schedule i hi
    rw [values i hi,if_pos covered]
    unfold point
    rw [direct_logical ops negate memory x domain.xLaws domain.admitted.contiguous.1 domain.admitted.noNeg.1 i hi,
      direct_logical ops negate memory y domain.yLaws domain.admitted.contiguous.2 domain.admitted.noNeg.2 i
        (by rw [←same_numel x.metadata y.metadata domain.admitted];exact hi)]
    exact formula i hi

#axiomsClean program_complete
#axiomsClean jobs_complete
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Kernel
