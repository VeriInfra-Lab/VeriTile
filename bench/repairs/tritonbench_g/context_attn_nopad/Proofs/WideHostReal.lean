import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideHostFormula

/-! Whole-grid adapter for the wider metadata domain. The successful kernel,
closed causal formula and finite flat-memory frame come from the unchanged
RealIO.implements_closed theorem. Generic grid merge/frame machinery is reused.
No source floating algorithm is copied or replaced. -/

namespace VeriTile.Bench.Repairs.ContextAttnNopad.WideHost
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Formula Program Host
noncomputable section
set_option maxHeartbeats 1600000
variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

theorem program_success (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
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

def frames (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    Kernel.GridFrames (A.flattenKernel (c.source r swapped).eraseDType.toAlgKernel)
      c.grid (A.flattenState s) := fun idx => {
  final := Classical.choose (program_success hm hd hcov swapped idx)
  writes := c.programWrites r A s idx
  h_exec := (Classical.choose_spec (program_success hm hd hcov swapped idx)).1
  h_writeWithin := (Classical.choose_spec (program_success hm hd hcov swapped idx)).2.2 }

theorem frames_disjoint (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    Kernel.GridWritesDisjoint (frames hm hd hcov swapped) := by
  intro idx idx' hne cell hi hj
  obtain ⟨_, i, hai, hei⟩ := (programWrites_iff c r A s idx cell).mp hi
  obtain ⟨_, j, haj, hej⟩ := (programWrites_iff c r A s idx' cell).mp hj
  have he : RealIO.write c.layout (c.launch idx) (inputs s r) i =
      RealIO.write c.layout (c.launch idx') (inputs s r) j :=
    Nat.add_left_cancel (hei.trans hej.symm)
  exact hne (unique_writer hm idx idx' i j hai haj he).1

/-- Whole-grid launch with all per-program successful executions and one merged
finite memory. No result equality or successful launch is assumed. -/
theorem grid_correct (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
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
variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

/-- A logical row is selected exactly up to the rounded host grid extent. -/
theorem covered (hm : WideDomain.Memory c r A s) (b h t d : Nat) (hb : b < c.batches)
    (hh : h < c.q.heads) (ht : t < c.length b) (hd : d < c.q.dims)
    (hg : t < c.blocks*128) :
    ∃ idx : GridIndex c.grid, ∃ i : RealIO.Lane c.layout,
      (c.launch idx).batch = b ∧ (c.launch idx).head = h ∧
      queryRow c.layout (c.launch idx) i.1.val = t ∧ i.2.1.val = d ∧
      RealIO.active c.layout (c.launch idx) (inputs s r) i := by
  have hblock : t / 128 < c.blocks := by omega
  let idx := c.index b h (t/128) hb hh hblock
  let i : RealIO.Lane c.layout := (⟨t%128, Nat.mod_lt _ (by decide)⟩, ⟨d, hd⟩, PUnit.unit)
  have htime : queryRow c.layout (c.launch idx) i.1.val = t := by
    simp only [idx, index_launch, queryRow, Call.layout, i]
    omega
  refine ⟨idx, i, rfl, rfl, htime, rfl, ?_⟩
  apply (active_iff hm idx i).mpr
  simpa only [htime] using ht

theorem writes_not_other_region (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (rg : RegionName) (hne : r.out ≠ rg) (off : Nat) (hoff : off < A.extent rg) :
    ¬ c.writes r A s (A.flat, A.addr rg off) := by
  rintro ⟨idx, hw⟩
  obtain ⟨_, i, hi, he⟩ := (programWrites_iff c r A s idx _).mp hw
  have hout := AddressBounds.inside_toNat ((bounds hm idx).output i.1 i.2.1 hi)
  change RealIO.write c.layout (c.launch idx) (inputs s r) i < A.extent r.out at hout
  have hmem : rg ∈ A.regions := by
    by_contra hn
    rw [hcov rg hn] at hoff
    omega
  have homem : r.out ∈ A.regions := by
    by_contra hn
    rw [hcov r.out hn] at hout
    omega
  have hd' := hd r.out homem rg hmem hne
  simp only [FlatAlloc.addr] at he
  rcases hd' with h | h <;> omega

theorem input_stable (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) {t : BlockState}
    (hf : BlockState.WriteWithin (c.writes r A s) (A.flattenState s) t) :
    ∀ rg ∈ c.inputRegions r, ∀ off, off < A.extent rg →
      t.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off) := by
  intro rg hrg off hoff
  have hne : r.out ≠ rg := by
    simp only [Call.inputRegions, List.mem_cons, List.not_mem_nil, or_false] at hrg
    rcases hrg with h | h | h | h | h
    · exact h ▸ hm.outQuery
    · exact h ▸ hm.outKey
    · exact h ▸ hm.outValue
    · exact h ▸ hm.outStart
    · exact h ▸ hm.outSequence
  exact (hf _ _ (writes_not_other_region hm hd hcov rg hne off hoff)).symm

/-- This theorem also covers insufficient `max_input_len`: only rows covered by
the rounded actual grid are claimed, and every unwritten cell is preserved. -/
theorem host_correct (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (swapped : Bool) :
    ∃ result, c.Returns r A s swapped result ∧
      (∀ b h t d, b < c.batches → h < c.q.heads → t < c.length b → d < c.q.dims →
        t < c.blocks*128 →
        result.1.readMem A.flat (A.addr r.out (c.output b h t d)) =
          c.expected (inputs s r) b h t d) ∧
      BlockState.WriteWithin (c.writes r A s) (A.flattenState s) result.1 ∧
      (∀ rg ∈ c.inputRegions r, ∀ off, off < A.extent rg →
        result.1.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off)) := by
  obtain ⟨out, hr, ho, hf⟩ := grid_correct hm hd hcov swapped
  refine ⟨(out, pythonReturn), ⟨hr, rfl⟩, ?_, hf, input_stable hm hd hcov hf⟩
  intro b h t d hb hh ht hdim hg
  obtain ⟨idx, i, hbatch, hhead, htime, hchannel, hi⟩ := covered hm b h t d hb hh ht hdim hg
  simpa only [hbatch, hhead, htime, hchannel] using ho idx i hi

/-- Sufficient max length yields every logical output element, with no padded
keys in its independent causal target. Empty batches and zero-length sequences
remain valid and simply have no output cells. -/
theorem host_complete (hm : WideDomain.Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0) (hc : c.completeGrid) (swapped : Bool) :
    ∃ result, c.Returns r A s swapped result ∧
      (∀ b h t d, b < c.batches → h < c.q.heads → t < c.length b → d < c.q.dims →
        result.1.readMem A.flat (A.addr r.out (c.output b h t d)) =
          c.expected (inputs s r) b h t d) ∧
      BlockState.WriteWithin (c.writes r A s) (A.flattenState s) result.1 ∧
      (∀ rg ∈ c.inputRegions r, ∀ off, off < A.extent rg →
        result.1.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off)) := by
  obtain ⟨result, hr, ho, hf, hs⟩ := host_correct hm hd hcov swapped
  refine ⟨result, hr, ?_, hf, hs⟩
  intro b h t d hb hh ht hdim
  exact ho b h t d hb hh ht hdim (lt_of_lt_of_le (lt_of_lt_of_le ht (hc b hb)) (blocks_cover c))

end
end VeriTile.Bench.Repairs.ContextAttnNopad.WideHost

#print axioms VeriTile.Bench.Repairs.ContextAttnNopad.WideHost.host_correct
