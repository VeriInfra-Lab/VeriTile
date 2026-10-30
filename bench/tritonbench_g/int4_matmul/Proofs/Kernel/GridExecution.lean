import bench.tritonbench_g.int4_matmul.Proofs.Kernel.Complete
import bench.tritonbench_g.int4_matmul.Proofs.Kernel.OperandBridge

namespace VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- Typed initial payloads; all memory outside the five kernel regions keeps
its original cells and dtype tags. -/
def memory (m : ScalarState α) (outside : RegionName → Nat → Cell α) (r : RegionName) (i : Nat) : Cell α :=
  if r="a" then .mk .real (m.values .a i)
  else if r="weight" then .mk .int (m.packed .weight i).toInt
  else if r="scale" then .mk .real (m.values .scale i)
  else if r="zero" then .mk .int (m.packed .zero i).toInt
  else if r="out" then .mk .real (m.values .output i)
  else outside r i

def attach (m : ScalarState α) (scratch : RegisterState α) : RegisterState α :=
  {scratch with mem:=memory m scratch.mem}

def atProgram (s : RegisterState α) (pid split : Nat) : RegisterState α :=
  {s with pids:=fun d => if d=0 then pid else if d=1 then split else s.pids d}

theorem attached_inputs [Inhabited α] (m : ScalarState α) (scratch : RegisterState α) (pid split : Nat) :
    InputBinding (atProgram (attach m scratch) pid split) m := by
  constructor <;> intro i <;> simp [atProgram,attach,memory]

/-- Execute the actual source up to its output epilogue. The proof constructs
successful execution; it does not assume the accumulator or its result. -/
def computed [Inhabited α] (o : Ops α) (rest : Algebra α) (p : Parameters) (s : RegisterState α) :
    {t : RegisterState α //
      run o rest (prelude p++[.forRange "k" 0 (steps p) 1 (body p)]) s=some t ∧
      Invariant o p s (steps p) t} := by
  let start := preludeExecution o rest p s
  let result := loopExecution o rest p s start.val 0 (Nat.zero_le _)
    (preludeInvariant o rest p s)
  have he := (Classical.choose_spec result).1
  have hi := (Classical.choose_spec result).2
  refine ⟨Classical.choose result,?_,hi⟩
  simp only [run_append,start.property,Option.bind_some]
  simpa [run,step] using he

def computedValues [Inhabited α] (o : Ops α) (rest : Algebra α) (p : Parameters) (s : RegisterState α) :
    CIndex p → α :=
  ((computed o rest p s).val.regs .real [p.blockRows,p.blockColumns] "accumulator").getD (fun _ => o.zero)

theorem computed_values [Inhabited α] (o : Ops α) (rest : Algebra α) (p : Parameters) (s : RegisterState α) :
    computedValues o rest p s=accumulation o p s (steps p) := by
  simp [computedValues,(computed o rest p s).property.2.acc]

/-- The source-derived lane commit operation is exactly the memory effect of
the full quoted program, including its selected store/atomic epilogue. -/
theorem source_program_events [Inhabited α] (o : Ops α) (rest : Algebra α)
    (p : Parameters) (s : RegisterState α) :
    ∃ final,exec o rest p s=some final ∧ final.mem=
      ((TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
        (writeLane o p s (computedValues o rest p s)) s).mem := by
  rw [computed_values]
  exact correctness_memory o rest p s

/-- The program prepares its load/compute DAG once. Its independent lane
commits may then be interleaved with other programs. Each value below is read
from execution of the quoted source, and each commit is the proven epilogue
operation. This is the atomic serialization model, not a numerical oracle. -/
def event [Inhabited α] (o : Ops α) (rest : Algebra α) (p : Parameters)
    (snapshot : RegisterState α) (l : Lane) (current : RegisterState α) : RegisterState α :=
  if h : l.i<p.blockRows ∧ l.j<p.blockColumns then
    let program := atProgram snapshot l.pid l.split
    writeLane o p program (computedValues o rest p program) current
      (⟨l.i,h.1⟩,⟨l.j,h.2⟩,PUnit.unit)
  else current

def parallel [Inhabited α] (o : Ops α) (rest : Algebra α) (p : Parameters)
    (snapshot : RegisterState α) (lanes : List Lane) : RegisterState α :=
  lanes.foldl (fun current l => event o rest p snapshot l current) snapshot

def laneSchedule (s : Shape) (t : Tile) (lanes : List Lane) : List Work :=
  (lanes.filter (fun l => decide (s.Active t (laneWork s t l)))).map (laneWork s t)

theorem schedule_complete (s : Shape) (t : Tile) (lanes : List Lane)
    (h : lanes.Perm (rawLanes s t)) : Complete s t (laneSchedule s t lanes) := by
  exact (h.filter _).map _

theorem memory_store (m : ScalarState α) (outside : RegionName → Nat → Cell α) (a : Nat) (v : α)
    (s : RegisterState α) (hm : s.mem=memory m outside) :
    (s.write "out" a (.mk .real v)).mem=
      memory (VeriTile.Bench.Repairs.Int4Matmul.store m .output a v) outside := by
  funext r i
  simp only [State.write,hm,memory,VeriTile.Bench.Repairs.Int4Matmul.store]
  split_ifs <;> simp_all

theorem event_memory [Inhabited α] (x : Host.Arguments) (hx : Host.Contract x) (t : Tile)
    (ht : t∈Host.sourceTiles) (o : Ops α) (rest : Algebra α)
    (initial current : ScalarState α) (scratch s : RegisterState α)
    (hm : s.mem=memory current scratch.mem) (l : Lane) (hl : l.Valid x.shape t) :
    (event o rest (parameters x t) (attach initial scratch) l s).mem=
      memory (if x.shape.Active t (laneWork x.shape t l) then
        Source.commit x.shape t o (laneWork x.shape t l)
          (partialValue x.shape t o initial (laneWork x.shape t l)) current else current) scratch.mem := by
  let program := atProgram (attach initial scratch) l.pid l.split
  let i : CIndex (parameters x t) := (⟨l.i,hl.2.2.1⟩,⟨l.j,hl.2.2.2⟩,PUnit.unit)
  have hw : work x t program i=laneWork x.shape t l := by
    cases l
    rfl
  have hv := source_partial x hx t ht o program initial (attached_inputs initial scratch l.pid l.split) i
  have ha : active (parameters x t) program i ↔ x.shape.Active t (laneWork x.shape t l) := by
    change (_ ∧ _) ↔ (_ ∧ _ ∧ _)
    simp only [←hw,←row_work,←column_work,split_work,program,atProgram]
    simp [hl.2.1]
    rfl
  have ho : outputOffset (parameters x t) program i=cAddress x.shape (laneWork x.shape t l).row (laneWork x.shape t l).col := by
    rw [←hw,←row_work,←column_work]
    simp [outputOffset,parameters,cAddress,Nat.mul_comm]
  simp only [event,parameters,hl.2.2.1,hl.2.2.2,and_self,↓reduceDIte]
  change (writeLane o (parameters x t) program (computedValues o rest (parameters x t) program) s i).mem=_
  rw [computed_values]
  unfold writeLane
  simp only [ha]
  by_cases live : x.shape.Active t (laneWork x.shape t l)
  · simp only [live,ite_true]
    rw [ho,hv,hw]
    have read : (s.mem "out" (cAddress x.shape (laneWork x.shape t l).row (laneWork x.shape t l).col)).read .real=
        current.values .output (cAddress x.shape (laneWork x.shape t l).row (laneWork x.shape t l).col) := by
      simp [hm,memory]
    simp only [parameters,read,Source.commit]
    exact memory_store current scratch.mem _ _ s hm
  · simpa only [live,ite_false] using hm

#axiomsClean computed
#axiomsClean computed_values
#axiomsClean source_program_events
#axiomsClean event_memory
end
end VeriTile.Bench.Repairs.Int4Matmul.KernelExecution
