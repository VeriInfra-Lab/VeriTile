import bench.tritonbench_g.context_attn_nopad.Proofs.HostFormula

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1200000

def Call.io (c : Call) (r : Regions) (swapped : Bool) : RealIO.IO c.layout :=
  RealIO.sourceIO c.layout r .fp32 swapped c.scale

def Call.programWrites (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (idx : GridIndex c.grid) : WriteFootprint :=
  (c.io r Bool.false).writes A (c.launch idx) (inputs s r)

theorem programWrites_iff (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (idx : GridIndex c.grid) (cell : MemCellAddr) :
    c.programWrites r A s idx cell ↔ cell.1 = A.flat ∧
      ∃ i, RealIO.active c.layout (c.launch idx) (inputs s r) i ∧
        A.addr r.out (RealIO.write c.layout (c.launch idx) (inputs s r) i) = cell.2 := by
  simp only [Call.programWrites, Call.io, RealIO.IO.writes, RealIO.IO.regionWrites, RealIO.sourceIO]
  constructor
  · rintro ⟨hf, rg, off, ⟨i, hi, rfl, rfl⟩, he⟩
    exact ⟨hf, i, hi, he⟩
  · rintro ⟨hf, i, hi, he⟩
    exact ⟨hf, r.out, _, ⟨i, hi, rfl, rfl⟩, he⟩

variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

theorem program_success (hm : Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool)
    (idx : GridIndex c.grid) :
    ∃ t, exec (A.flattenKernel (c.source r swapped).eraseDType.toAlgKernel)
        ((A.flattenState s).withGridIndex idx) = some t ∧
      (∀ i, RealIO.active c.layout (c.launch idx) (inputs s r) i →
        t.readMem A.flat (A.addr r.out (RealIO.write c.layout (c.launch idx) (inputs s r) i)) =
          c.expected (inputs s r) (c.launch idx).batch (c.launch idx).head
            (queryRow c.layout (c.launch idx) i.1.val) i.2.1.val) ∧
      BlockState.WriteWithin (c.programWrites r A s idx) ((A.flattenState s).withGridIndex idx) t := by
  have hv : RealIO.valid c.layout (Program.launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) r) := by
    rw [launch_grid, inputs_grid]
    exact valid hm idx
  have hb : AddressBounds.Valid c.layout r (Program.launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) r) A.extent := by
    rw [launch_grid, inputs_grid]
    exact bounds hm idx
  obtain ⟨t, he, ho, hf⟩ := RealIO.implements_closed c.layout r .fp32 swapped c.scale
    (by norm_num [Call.layout]) A hd hcov (s.withGridIndex idx) hv hb
  refine ⟨t, he, ?_, ?_⟩
  · intro i hi
    have hout := ho i hi
    change t.readMem A.flat (A.addr r.out (RealIO.write c.layout (Program.launch (s.withGridIndex idx))
      (inputs (s.withGridIndex idx) r) i)) =
      RealIO.closedExpected c.layout c.scale (Program.launch (s.withGridIndex idx)) (inputs s r) i at hout
    rw [launch_grid, inputs_grid, closedExpected_logical hm idx i hi] at hout
    exact hout
  · simpa only [RealIO.sourceIO, launch_grid, inputs_grid, Call.programWrites, Call.io] using hf

def frames (hm : Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    Kernel.GridFrames (A.flattenKernel (c.source r swapped).eraseDType.toAlgKernel)
      c.grid (A.flattenState s) := fun idx => {
  final := Classical.choose (program_success hm hd hcov swapped idx)
  writes := c.programWrites r A s idx
  h_exec := (Classical.choose_spec (program_success hm hd hcov swapped idx)).1
  h_writeWithin := (Classical.choose_spec (program_success hm hd hcov swapped idx)).2.2 }

theorem frames_disjoint (hm : Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    Kernel.GridWritesDisjoint (frames hm hd hcov swapped) := by
  intro idx idx' hne cell hi hj
  obtain ⟨_, i, hai, hei⟩ := (programWrites_iff c r A s idx cell).mp hi
  obtain ⟨_, j, haj, hej⟩ := (programWrites_iff c r A s idx' cell).mp hj
  have he : RealIO.write c.layout (c.launch idx) (inputs s r) i =
      RealIO.write c.layout (c.launch idx') (inputs s r) j :=
    Nat.add_left_cancel (hei.trans hej.symm)
  exact hne (unique_writer hm idx idx' i j hai haj he).1

def Call.writes (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) : WriteFootprint :=
  fun cell => ∃ idx : GridIndex c.grid, c.programWrites r A s idx cell

/-- Whole-grid launch with all per-program successful executions and one merged
finite memory. No result equality or successful launch is assumed. -/
theorem grid_correct (hm : Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    ∃ t, Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel (c.source r swapped).eraseDType.toAlgKernel) c.grid (A.flattenState s) t) ∧
      (∀ idx : GridIndex c.grid, ∀ i,
        RealIO.active c.layout (c.launch idx) (inputs s r) i →
        t.readMem A.flat (A.addr r.out (c.output (c.launch idx).batch (c.launch idx).head
          (queryRow c.layout (c.launch idx) i.1.val) i.2.1.val)) =
        c.expected (inputs s r) (c.launch idx).batch (c.launch idx).head
          (queryRow c.layout (c.launch idx) i.1.val) i.2.1.val) ∧
      BlockState.WriteWithin (c.writes r A s) (A.flattenState s) t := by
  let fs := frames hm hd hcov swapped
  refine ⟨Kernel.mergeFrames c.grid (A.flattenState s) fs,
    ⟨⟨fs, frames_disjoint hm hd hcov swapped, rfl⟩⟩, ?_, Kernel.mergeFrames_writeWithin fs⟩
  intro idx i hi
  have hw : (fs idx).writes (A.flat, A.addr r.out (RealIO.write c.layout (c.launch idx) (inputs s r) i)) :=
    (programWrites_iff c r A s idx _).mpr ⟨rfl, i, hi, rfl⟩
  have he := Kernel.mergeFrames_mem_written (frames_disjoint hm hd hcov swapped) idx _ _ hw
  rw [← write_eq hm idx i]
  unfold BlockState.readMem
  rw [he]
  exact (Classical.choose_spec (program_success hm hd hcov swapped idx)).2.1 i hi

end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
