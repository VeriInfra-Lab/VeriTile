import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FPTotal

namespace VeriTile.Bench.Repairs.BmmChunkFwd.FPGrid
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section
set_option maxHeartbeats 2000000

/-- Actual grid coordinates with fresh register state for every program. -/
def atIndex {α : Type} {g : Grid} (s : State α) (idx : GridIndex g) : State α :=
  {s with regs:=fun _ _ _ => none, pids:=idx.toPids, numPids:=g.toNumPids}

def schedule {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call) (c : Host.Config) :
    List (GridIndex (h.grid c)) → State α → Option (State α)
  | [],s => some s
  | idx::rest,s => do
    let next ← FP.Structural.exec (engine o.algebra) (Kernels.source h c) (atIndex s idx)
    schedule o h c rest next

def allPrograms (h : Host.Call) (c : Host.Config) : List (GridIndex (h.grid c)) := Finset.univ.toList

theorem allPrograms_complete (h : Host.Call) (c : Host.Config) (idx : GridIndex (h.grid c)) :
    idx∈allPrograms h c := by simp [allPrograms]

def written {α : Type} (h : Host.Call) (c : Host.Config) (order : List (GridIndex (h.grid c)))
    (s : State α) (rg : RegionName) (off : Nat) : Prop :=
  ∃ idx∈order, FPSuccess.ProgramWrites h c (atIndex s idx) rg off

/-- Successful whole typed grid, with all programs and their memory effects.
No source-execution premise occurs: each full program is constructed from the
actual prologue, arbitrary local dot results, K loop and epilogue. -/
theorem successful_schedule {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (order : List (GridIndex (h.grid c))) (s : State α) :
    ∃ t, schedule o h c order s=some t ∧
      (∀ rg off, ¬written h c order s rg off → t.mem rg off=s.mem rg off) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ => rfl⟩
  | cons idx rest ih =>
    obtain ⟨mid,hm,hframe,_⟩ := FPSuccess.complete o h c (atIndex s idx)
    obtain ⟨final,hf,hff⟩ := ih mid
    refine ⟨final,?_,?_⟩
    · simp only [schedule,hm,Option.bind_some]
      exact hf
    · intro rg off hrg
      rw [hff rg off (by
        rintro ⟨i,hi,hw⟩
        exact hrg ⟨i,List.mem_cons_of_mem idx hi,hw⟩)]
      exact hframe rg off (fun hw => hrg ⟨idx,List.mem_cons_self,hw⟩)

theorem schedule_region_frame {α : Type} [Inhabited α] (h : Host.Call) (c : Host.Config)
    (order : List (GridIndex (h.grid c))) (s t : State α)
    (frame : ∀ rg off, ¬written h c order s rg off → t.mem rg off=s.mem rg off) :
    FPSuccess.RegionFrame h.Out s t := by
  intro rg off hrg
  apply frame
  rintro ⟨idx,_,_,i,_,he,_⟩
  exact hrg he

def hostWritten {α : Type} (h : Host.Call) (c : Host.Config) (s : State α) (rg : RegionName) (off : Nat) : Prop :=
  h.shape.outputSize≠0 ∧ written h c (allPrograms h c) s rg off

/-- Exact wrapper no-launch behavior for empty outputs, including zero groups. -/
def runHost {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call) (c : Host.Config)
    (s : State α) : Option (State α) :=
  if h.shape.outputSize=0 then some s else schedule o h c (allPrograms h c) s

theorem successful_host {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (s : State α) :
    ∃ final, runHost o h c s=some final ∧ FPSuccess.RegionFrame h.Out s final ∧
      (h.shape.outputSize=0 → final=s) ∧
      (∀ rg off, ¬hostWritten h c s rg off → final.mem rg off=s.mem rg off) := by
  by_cases empty : h.shape.outputSize=0
  · exact ⟨s,by simp [runHost,empty],fun _ _ _ => rfl,fun _ => rfl,fun _ _ _ => rfl⟩
  · obtain ⟨final,hf,hframe⟩ := successful_schedule o h c (allPrograms h c) s
    exact ⟨final,by simpa [runHost,empty] using hf,
      schedule_region_frame h c _ s final hframe,by simp [empty],
      fun rg off hmiss => hframe rg off (fun hw => hmiss ⟨empty,hw⟩)⟩

/-- Independent original logical arrays in the floating carrier, including
payloads and signed sequence IDs from the original direct buffers. -/
structure LogicalInputs (α : Type) where
  a : Nat → Nat → Nat → Nat → α
  b : Nat → Nat → Nat → Nat → α
  seq : Nat → Nat → Int

def returnedOffset (h : Host.Call) (b ch g r n : Nat) : Nat :=
  ((b*h.shape.chunks+ch)*h.shape.groups+g)*h.shape.chunk*h.shape.chunk+r*h.shape.chunk+n

def readback {α : Type} [Inhabited α] (h : Host.Call) (s : State α) (b ch g r n : Nat) : α :=
  toFloat (Host.outputDType h.aStorage h.requestedOutput)
    ((s.mem h.Out (returnedOffset h b ch g r n)).read
      (Host.outputDType h.aStorage h.requestedOutput).toTileDType)

/-- A returned tensor object observes the actual final typed memory, with
identity, shape and dtype from the same fresh allocation. -/
structure Returned (α : Type) (h : Host.Call) where
  region : RegionName
  shape : List Nat
  dtype : FloatDType
  value : Nat → Nat → Nat → Nat → Nat → α

def returned {α : Type} [Inhabited α] (h : Host.Call) (s : State α) : Returned α h :=
  ⟨h.Out,h.shape.outputShape,Host.outputDType h.aStorage h.requestedOutput,readback h s⟩

/-- Complete typed reference execution and direct returned-allocation readback.
The existential result is constructed, not assumed. Floating values remain
opaque; accuracy against the independent Real contraction is a separate local
primitive numerical obligation. -/
theorem returned_grid {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (s : State α) :
    ∃ final, ∃ result : Returned α h,
      runHost o h c s=some final ∧ result=returned h final ∧
      result.region=h.Out ∧ result.shape=h.shape.outputShape ∧
      result.dtype=Host.outputDType h.aStorage h.requestedOutput ∧
      (∀ b ch g r n, result.value b ch g r n=readback h final b ch g r n) ∧
      FPSuccess.RegionFrame h.Out s final ∧ (h.shape.outputSize=0 → final=s) ∧
      (∀ rg off, ¬hostWritten h c s rg off → final.mem rg off=s.mem rg off) := by
  obtain ⟨final,hf,hframe,hempty,hfull⟩ := successful_host o h c s
  exact ⟨final,returned h final,hf,rfl,rfl,rfl,rfl,fun _ _ _ _ _ => rfl,hframe,hempty,hfull⟩

#axiomsClean successful_schedule
#axiomsClean returned_grid
end
end VeriTile.Bench.Repairs.BmmChunkFwd.FPGrid

#guard_msgs (drop info) in
#auditModuleAxioms
