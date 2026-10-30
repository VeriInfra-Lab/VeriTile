import bench.tritonbench_g.context_attn_nopad.Proofs.Mathematical_HostGrid

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1000000

def Call.index (c : Call) (b h block : Nat) (hb : b < c.batches)
    (hh : h < c.q.heads) (ht : block < c.blocks) : GridIndex c.grid := by
  rintro ⟨ax, ha⟩
  rcases ax with _ | (_ | (_ | ax))
  · exact ⟨b, hb⟩
  · exact ⟨h, hh⟩
  · exact ⟨block, ht⟩
  · have : ax + 3 < 3 := ha
    omega

@[simp] theorem index_launch (c : Call) (b h block : Nat) (hb : b < c.batches)
    (hh : h < c.q.heads) (ht : block < c.blocks) :
    c.launch (c.index b h block hb hh ht) = ⟨block, b, h⟩ := rfl

theorem blocks_cover (c : Call) : c.maxInputLen ≤ c.blocks * 128 := by
  unfold Call.blocks
  omega

variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

/-- A logical row is selected exactly up to the rounded host grid extent. -/
theorem covered (hm : Memory c r A s) (b h t d : Nat) (hb : b < c.batches)
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

theorem writes_not_other_region (hm : Memory c r A s) (hd : A.Disjoint)
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

def Call.inputRegions (_c : Call) (r : Regions) : List RegionName :=
  [r.query, r.key, r.value, r.start, r.sequence]

theorem input_stable (hm : Memory c r A s) (hd : A.Disjoint)
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

/-- The wrapper's execution boundary: the complete actual grid is launched over
the caller-owned finite buffers, and Python returns `None`. -/
def Call.Returns (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) (swapped : Bool)
    (result : BlockState × Option Unit) : Prop :=
  Nonempty (Kernel.GridLaunchedOrdinary (A.flattenKernel (c.source r swapped).eraseDType.toAlgKernel)
    c.grid (A.flattenState s) result.1) ∧ result.2 = pythonReturn

/-- This theorem also covers insufficient `max_input_len`: only rows covered by
the rounded actual grid are claimed, and every unwritten cell is preserved. -/
theorem host_correct (hm : Memory c r A s) (hd : A.Disjoint)
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
theorem host_complete (hm : Memory c r A s) (hd : A.Disjoint)
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
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
