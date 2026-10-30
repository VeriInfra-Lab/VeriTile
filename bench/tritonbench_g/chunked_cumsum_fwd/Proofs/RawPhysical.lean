import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawObservedIO

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
open VeriTile.Triton FP.Structural Math Program RawObserved Backend RawBackend
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

abbrev PhysicalMemory (α : Type) := Nat → Nat → α

structure View where
  allocation : Nat
  base : Nat
  deriving DecidableEq

def View.read (v : View) (memory : PhysicalMemory α) (off : Nat) : α :=
  memory v.allocation (v.base+off)

/-- Physical allocation views. Read-only inputs may share allocations or
overlap; each output allocation is fresh and distinct. No value is supplied. -/
structure Views where
  dt : View
  rate : View
  bias : View
  out : View
  sum : View
  distinct : out.allocation ≠ sum.allocation

def ports : Regions := ⟨"dt","A","bias","out","sum"⟩
theorem ports_fresh : Whole.Fresh ports := by constructor <;> decide

def Views.words (v : Views) (memory : PhysicalMemory α) : InputWords α :=
  ⟨v.dt.read memory,v.rate.read memory,v.bias.read memory⟩

def Views.state (v : Views) (memory : PhysicalMemory α) : State (Term α) where
  mem := fun rg off =>
    if rg = ports.dt then Cell.mk .fp32 (.input (v.dt.read memory off))
    else if rg = ports.rate then Cell.mk .fp32 (.input (v.rate.read memory off))
    else if rg = ports.bias then Cell.mk .real (.input (v.bias.read memory off))
    else if rg = ports.out then Cell.mk .real (.input (v.out.read memory off))
    else Cell.mk .real (.input (v.sum.read memory off))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

theorem state_reads (v : Views) (memory : PhysicalMemory α) :
    InputGraph ports (v.state memory) (v.words memory) := by
  refine ⟨?_,?_,?_⟩ <;> intro off <;> simp [Views.state,Views.words,ports]

def InView (v : View) (n allocation off : Nat) : Prop :=
  allocation = v.allocation ∧ v.base ≤ off ∧ off < v.base+n

def Views.written (v : Views) (c : Whole.Host) (allocation off : Nat) : Prop :=
  InView v.out c.outputSize allocation off ∨ InView v.sum c.outputSize allocation off

/-- Commit only the two full output ranges from the actual source result.
No formula is substituted into memory. -/
def Views.commit (v : Views) (M : NativeModel α) (c : Whole.Host)
    (memory : PhysicalMemory α) (final : State (Term α)) : PhysicalMemory α :=
  fun allocation off =>
    if InView v.out c.outputSize allocation off then
      interpret M ((final.mem ports.out (off-v.out.base)).read .real)
    else if InView v.sum c.outputSize allocation off then
      interpret M ((final.mem ports.sum (off-v.sum.base)).read .real)
    else memory allocation off

theorem commit_out (v : Views) (M : NativeModel α) (c : Whole.Host)
    (memory : PhysicalMemory α) (final : State (Term α)) (off : Nat) (h : off < c.outputSize) :
    v.out.read (v.commit M c memory final) off = interpret M ((final.mem ports.out off).read .real) := by
  simp [View.read,Views.commit,InView,h]

theorem commit_sum (v : Views) (M : NativeModel α) (c : Whole.Host)
    (memory : PhysicalMemory α) (final : State (Term α)) (off : Nat) (h : off < c.outputSize) :
    v.sum.read (v.commit M c memory final) off = interpret M ((final.mem ports.sum off).read .real) := by
  simp [View.read,Views.commit,InView,h,Ne.symm v.distinct]

theorem commit_frame (v : Views) (M : NativeModel α) (c : Whole.Host)
    (memory : PhysicalMemory α) (final : State (Term α)) (allocation off : Nat)
    (outside : ¬ v.written c allocation off) :
    v.commit M c memory final allocation off = memory allocation off := by
  have h := not_or.mp outside
  simp only [Views.commit,if_neg h.1,if_neg h.2]

theorem packed_bound (c : Whole.Host) (b h k j : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hk : k < c.chunks) (hj : j < c.chunk) :
    Whole.packed c b h k j < c.outputSize :=
  Whole.pair_bound (Whole.pair_bound (Whole.pair_bound hb hh) hk) hj

def PhysicalValues (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (v : Views) (a : InputWords α) (hasBias useSoftplus : Bool) (lower upper : α)
    (memory : PhysicalMemory α) : Prop :=
  ∀ b h k j, b < c.batches → h < c.heads → k < c.chunks → ∀ hj : j < c.chunk,
    v.out.read memory (Whole.packed c b h k j) =
      referenceLogicalPrepared M c a hasBias useSoftplus lower upper b h k j ∧
    v.sum.read memory (Whole.packed c b h k j) =
      referenceLogicalCumulative M c plan a hasBias useSoftplus lower upper b h k
        ⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩

def runPhysical (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (v : Views) (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (memory : PhysicalMemory α) : Option (PhysicalMemory α) :=
  (RawObserved.runHost M c plan ports hasBias useSoftplus lower upper ls (v.state memory)).map
    (fun result => v.commit M c memory result.graph)

theorem physical_correct (M : NativeModel α) (c : Whole.Host) (plan : PrefixPlan c.BH c.BC)
    (v : Views) (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (schedule : ls.Perm c.grid) (memory : PhysicalMemory α) :
    ∃ final, runPhysical M c plan v hasBias useSoftplus lower upper ls memory = some final ∧
      PhysicalValues M c plan v (v.words memory) hasBias useSoftplus lower upper final ∧
      (∀ allocation off, ¬ v.written c allocation off → final allocation off = memory allocation off) := by
  obtain ⟨result,ran,_,_,_,values,_⟩ := RawObserved.host_correct M c plan ports ports_fresh
    (v.state memory) (v.words memory) (state_reads v memory) hasBias useSoftplus lower upper ls schedule
  refine ⟨v.commit M c memory result.graph,?_,?_,commit_frame v M c memory result.graph⟩
  · simp only [runPhysical,ran,Option.map_some]
  · intro b h k j hb hh hk hj
    rw [commit_out v M c memory result.graph _ (packed_bound c b h k j hb hh hk hj),
      commit_sum v M c memory result.graph _ (packed_bound c b h k j hb hh hk hj)]
    exact values b h k j hb hh hk hj

def FreshPhysical (v : Views) (preexisting : Nat → Prop) : Prop :=
  preexisting v.dt.allocation ∧ preexisting v.rate.allocation ∧ preexisting v.bias.allocation ∧
  ¬ preexisting v.out.allocation ∧ ¬ preexisting v.sum.allocation

theorem fresh_outside (v : Views) (c : Whole.Host) (fresh : FreshPhysical v preexisting)
    (allocation off : Nat) (old : preexisting allocation) : ¬ v.written c allocation off := by
  rintro (h | h)
  · exact fresh.2.2.2.1 (h.1 ▸ old)
  · exact fresh.2.2.2.2 (h.1 ▸ old)

theorem words_frame (v : Views) (fresh : FreshPhysical v preexisting)
    (before after : PhysicalMemory α)
    (frame : ∀ allocation off, preexisting allocation → after allocation off = before allocation off) :
    v.words after = v.words before := by
  cases v with | mk dt rate bias out sum distinct =>
    simp only [Views.words,View.read]
    congr 1 <;> funext off
    · exact frame _ _ fresh.1
    · exact frame _ _ fresh.2.1
    · exact frame _ _ fresh.2.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBoundary
