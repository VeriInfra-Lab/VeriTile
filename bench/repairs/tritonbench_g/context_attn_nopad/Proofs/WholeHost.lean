import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Logical

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Logical
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
noncomputable section
set_option maxHeartbeats 1200000

/-- Exact independently stated logical write footprint, including rounded
under-dispatch. No execution/result occurs in this predicate. -/
def Written (c : Call) (i : Coord c) : Prop :=
  ∃ b t, b < c.batches ∧ t < c.length b ∧ t < c.blocks*128 ∧
    i.1.val = c.start b+t ∧ i.2.1.val < c.q.heads ∧ i.2.2.val < c.q.dims

def outCoord {c : Call} (hs : Shape c) (b h t d : Nat)
    (hb : b < c.batches) (hh : h < c.q.heads) (ht : t < c.length b)
    (hd : d < c.q.dims) : Coord c :=
  (⟨c.start b+t, by have := hs.oRows b hb; omega⟩,
   ⟨h, lt_of_lt_of_le hh hs.oHeads⟩, ⟨d, lt_of_lt_of_le hd hs.oDims⟩)

theorem writes_implies_written {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (i : Coord c)
    (hw : c.writes r A s (A.flat, A.addr r.out (coordOffset c i))) : Written c i := by
  obtain ⟨idx, hw⟩ := hw
  obtain ⟨_, lane, hlane, he⟩ := (programWrites_iff c r A s idx _).mp hw
  have he' : VeriTile.Bench.TritonBenchG.ContextAttnNopad.RealIO.write c.layout (c.launch idx) (inputs s r) lane =
      coordOffset c i := Nat.add_left_cancel he
  rw [write_eq hm] at he'
  obtain ⟨hrow, hhead, hdim⟩ := hm.shape.outputInjective _ _ _ _ _ _
    (logical_row_lt hm idx lane hlane) i.1.isLt
    (lt_of_lt_of_le (launch_head_lt c idx) hm.shape.oHeads) i.2.1.isLt
    (lt_of_lt_of_le lane.2.1.isLt hm.shape.oDims) i.2.2.isLt he'
  refine ⟨(c.launch idx).batch, queryRow c.layout (c.launch idx) lane.1.val,
    launch_batch_lt c idx, (active_iff hm idx lane).mp hlane, ?_,
    hrow.symm, ?_, ?_⟩
  · have hb := launch_block_lt c idx
    have hi : lane.1.val < 128 := lane.1.isLt
    dsimp [queryRow, Call.layout]
    omega
  · rw [← hhead]; exact launch_head_lt c idx
  · rw [← hdim]; exact lane.2.1.isLt

def Values (c : Call) (r : Regions) (A : FlatAlloc) (t : BlockState) : Coord c → ℝ :=
  fun i => t.readMem A.flat (A.addr r.out (coordOffset c i))

/-- Initialization stated at the flat allocation boundary used by the reused
execution theorem. It is the ordinary initialized logical-copy contract. -/
def FlatInitialized (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (view : OutputView c) (before : Nat → ℝ) : Prop :=
  ∀ i, Values c r A (A.flattenState s) i = view.read before i

def Contract (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (hs : Shape c) (a : Inputs) (view : OutputView c) (before : Nat → ℝ) : Prop :=
  ∃ raw, c.Returns r A s Bool.false raw ∧
    let result := finish view before (Values c r A raw.1)
    result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧
    result.pythonReturn = none ∧
    (∀ b h t d hb hh ht hd, t < c.blocks*128 →
      view.read result.memory (outCoord hs b h t d hb hh ht hd) = expected c a b h t d) ∧
    (∀ addr, (¬ ∃ i, Written c i ∧ view.address i = addr) → result.memory addr = before addr) ∧
    BlockState.WriteWithin (c.writes r A s) (A.flattenState s) raw.1 ∧
    (∀ rg ∈ c.inputRegions r, ∀ off, off < A.extent rg →
      raw.1.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off))

/-- Complete real host/grid composition: setup maps every logical input
coordinate, the original full-grid theorem supplies successful finite execution
and frames, and coordinate writeback restores the original output allocation. -/
theorem whole_host_correct {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (a : Inputs) (hn : Normalizes c (inputs s r) a)
    (view : OutputView c) (hinj : Function.Injective view.address)
    (before : Nat → ℝ) (hi : FlatInitialized c r A s view before) :
    Contract c r A s hm.shape a view before := by
  obtain ⟨raw, hr, ho, hf, hs⟩ := host_correct hm hd hcov Bool.false
  refine ⟨raw, hr, rfl, rfl, rfl, ?_, ?_, hf, hs⟩
  · intro b h t d hb hh ht hdim hg
    dsimp only [finish]
    rw [writeback_read view hinj]
    change raw.1.readMem A.flat (A.addr r.out (c.output b h t d)) = _
    rw [ho b h t d hb hh ht hdim hg]
    exact expected_normalizes hm.shape hn b h t d hb hh ht hdim
  · intro addr hout
    apply writeback_frame view before (Values c r A raw.1) (Written c) _ addr hout
    intro i hnot
    have hnw : ¬ c.writes r A s (A.flat, A.addr r.out (coordOffset c i)) :=
      fun hw => hnot (writes_implies_written hm i hw)
    have he := hf A.flat (A.addr r.out (coordOffset c i)) hnw
    change raw.1.readMem A.flat (A.addr r.out (coordOffset c i)) = _
    unfold BlockState.readMem
    rw [← he]
    exact hi i

/-- Every output coordinate is covered when max_input_len covers lengths;
there is no padded-key or block-count dependency in its mathematical target. -/
theorem complete_readback {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hs : Shape c) {a : Inputs} {view : OutputView c} {before : Nat → ℝ}
    (hc : c.completeGrid) (h : Contract c r A s hs a view before) :
    ∃ raw, c.Returns r A s Bool.false raw ∧
      ∀ b head t d hb hh ht hd,
        view.read (finish view before (Values c r A raw.1)).memory
          (outCoord hs b head t d hb hh ht hd) = expected c a b head t d := by
  obtain ⟨raw, hr, _, _, _, ho, _⟩ := h
  refine ⟨raw, hr, ?_⟩
  intro b head t d hb hh ht hd
  apply ho b head t d hb hh ht hd
  exact lt_of_lt_of_le (lt_of_lt_of_le ht (hc b hb)) (blocks_cover c)

end
end VeriTile.Bench.Repairs.ContextAttnNopad.Logical
