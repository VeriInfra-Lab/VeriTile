import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPHostSource
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPFrame
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.HostSource

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPHostPrecise
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula Host HostPlan FPHostSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

theorem launch_eq {α : Type} (c : Call) (ha : NativeProgram.ShapeABI c)
    (hb : c.blocks ≤ 16777216) (j : Job c) (s : State α) :
    FPSymbolic.launchOf (atProgram c j s) = c.launch (index c j) := by
  rw [atProgram_eq c ha hb]
  simp [FPSymbolic.launchOf,FPHost.atProgram,Call.launch,GridIndex.toPids,Call.grid,Grid.rank]

/-- Reuse the already checked source write/mask formulas and original flat
footprint. This connects an opaque-FP store to the independent host footprint. -/
theorem writes_subset {α : Type} [Inhabited α]
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State α}
    (hm : Memory c r A layoutState) (ha : NativeProgram.ABI c A) (hi : Metadata c r s)
    (hb : c.blocks ≤ 16777216) (j : Job c) (rg : RegionName) (off : Nat)
    (hw : FPSymbolic.Writes c.layout r (atProgram c j s) rg off) :
    c.writes r A layoutState (A.flat,A.addr rg off) := by
  obtain ⟨lane,hactive,hrg,hoff⟩ := hw
  rw [launch_eq c (NativeProgram.shapeABI hm ha) hb j s] at hactive hoff
  have hbatch := launch_batch_lt c (index c j)
  have hstart : (FPSymbolic.integerInputs r (atProgram c j s)).start (c.launch (index c j)).batch =
      (Program.inputs layoutState r).start (c.launch (index c j)).batch :=
    (hi.start _ hbatch).trans (hm.start _ hbatch).symm
  have hlength : (FPSymbolic.integerInputs r (atProgram c j s)).sequence (c.launch (index c j)).batch =
      (Program.inputs layoutState r).sequence (c.launch (index c j)).batch :=
    (hi.length _ hbatch).trans (hm.length _ hbatch).symm
  have hactive' : RealIO.active c.layout (c.launch (index c j)) (Program.inputs layoutState r) lane := by
    simpa only [RealIO.active,Formula.sequence,hlength] using hactive
  have hoff' : RealIO.write c.layout (c.launch (index c j)) (Program.inputs layoutState r) lane = off := by
    simpa only [RealIO.write,AddressBounds.outputAddress,hstart] using hoff
  refine ⟨index c j,(programWrites_iff c r A layoutState (index c j) _).mpr ⟨rfl,lane,hactive',?_⟩⟩
  rw [hoff',hrg]

theorem program_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State α}
    (hm : Memory c r A layoutState) (ha : NativeProgram.ABI c A) (hi : Metadata c r s)
    (hb : c.blocks ≤ 16777216) (j : Job c) :
    ∃ t, FP.Structural.exec (FP.ProfiledRewrite.engine o.algebra) (kernel c r)
        (atProgram c j s) = some t ∧ Metadata c r t ∧
      (∀ rg off, ¬ c.writes r A layoutState (A.flat,A.addr rg off) → t.mem rg off = s.mem rg off) := by
  obtain ⟨t,he,hframe⟩ := FPSymbolic.complete_precise o c.layout r c.scale
    (by change 0 < (128 : Nat); omega) (atProgram c j s)
  refine ⟨t,he,?_,?_⟩
  · constructor
    · intro b hb'
      have hf := hframe r.start b (by
        rintro ⟨_,_,h,_⟩
        exact hm.outStart h)
      rw [hf]
      exact hi.start b hb'
    · intro b hb'
      have hf := hframe r.sequence b (by
        rintro ⟨_,_,h,_⟩
        exact hm.outSequence h)
      rw [hf]
      exact hi.length b hb'
  · intro rg off hnot
    exact hframe rg off (fun hw => hnot (writes_subset hm ha hi hb j rg off hw))

/-- Successful complete native stream with the exact host write footprint,
including cells in the output region. Metadata stability is proved at each step. -/
theorem programs_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (hb : c.blocks ≤ 16777216) (order : List (Job c)) (s : State α) (hi : Metadata c r s) :
    ∃ t, runPrograms o.algebra c r order s = some t ∧ Metadata c r t ∧
      (∀ rg off, ¬ c.writes r A layoutState (A.flat,A.addr rg off) → t.mem rg off = s.mem rg off) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,hi,by intros; rfl⟩
  | cons j rest ih =>
    obtain ⟨mid,he,hmeta,hframe⟩ := program_precise o hm ha hi hb j
    obtain ⟨t,ht,hmetat,hframet⟩ := ih mid hmeta
    refine ⟨t,?_,hmetat,?_⟩
    · simp only [runPrograms,he,Option.bind_some,ht]
    · intro rg off hnot
      exact (hframet rg off hnot).trans (hframe rg off hnot)

/-- Ordinary initialized resolve_neg scratch contract on opaque FP values. -/
def Initialized {α : Type} [Inhabited α] (c : Call) (r : Regions) (s : State α)
    (v : OutputView α c) (before : Nat → α) : Prop :=
  ∀ i, (s.mem r.out (Logical.coordOffset c i)).read .real = v.action (before (v.address i))

theorem copyback_precise {α : Type} [Inhabited α] {c : Call} (r : Regions)
    (v : OutputView α c) (before : Nat → α) (raw : State α)
    (stable : ∀ i, ¬ Logical.Written c i →
      (raw.mem r.out (Logical.coordOffset c i)).read .real = v.action (before (v.address i)))
    (addr : Nat) (hout : ¬ ∃ i, Logical.Written c i ∧ v.address i = addr) :
    (finish r v before raw).memory addr = before addr := by
  dsimp only [finish,writeback]
  split
  next h =>
    have hn : ¬ Logical.Written c (Classical.choose h) := by
      intro hw
      exact hout ⟨_,hw,Classical.choose_spec h⟩
    rw [stable _ hn,v.involution,Classical.choose_spec h]
  next => rfl

def Contract {α : Type} [Inhabited α] (M : Algebra α)
    (c : Call) (r : Regions) (A : FlatAlloc) (layoutState : BlockState) (s : State α)
    (v : OutputView α c) (before : Nat → α) : Prop :=
  ∃ raw result,
    runPrograms M (FPHostSource.effective c r s) r (HostPlan.jobs (FPHostSource.effective c r s)) s = some raw ∧
    runHost M c r s v before = some result ∧ reference M c r s v before = some result ∧
    result = finish r v before raw ∧
    result.objectId = v.objectId ∧ result.allocationId = v.allocationId ∧ result.pythonReturn = none ∧
    (∀ i, v.action (result.memory (v.address i)) = (raw.mem r.out (Logical.coordOffset c i)).read .real) ∧
    (∀ addr, (¬ ∃ i, Logical.Written c i ∧ v.address i = addr) → result.memory addr = before addr) ∧
    (∀ rg off, ¬ (FPHostSource.effective c r s).writes r A layoutState (A.flat,A.addr rg off) →
      raw.mem rg off = s.mem rg off)

theorem whole_host_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State α}
    (hm : Memory c r A layoutState) (ha : NativeProgram.ABI c A) (hi : Metadata c r s)
    (v : OutputView α c) (before : Nat → α) (hinit : Initialized c r s v before) :
    Contract o.algebra c r A layoutState s v before := by
  have he : FPHostSource.effective c r s = HostPlan.effective c r layoutState := by
    unfold FPHostSource.effective HostPlan.effective
    rw [FPHostSource.metadataMaximum_eq hi,HostPlan.metadataMaximum_eq hm]
  have hm' : Memory (FPHostSource.effective c r s) r A layoutState := by
    rw [he]
    exact effective_memory hm
  have hi' : Metadata (FPHostSource.effective c r s) r s := ⟨hi.start,hi.length⟩
  obtain ⟨raw,hr,_,hframe⟩ := programs_precise o hm' ⟨ha.heads,ha.bytes⟩
    (FPHostSource.blocks_bound hm.shape hi) (HostPlan.jobs _) s hi'
  have hh : runHost o.algebra c r s v before = some (finish r v before raw) := by
    dsimp only [runHost]
    rw [hr]
    rfl
  refine ⟨raw,finish r v before raw,hr,hh,?_,rfl,rfl,rfl,rfl,caller_readback r v before raw,?_,hframe⟩
  · rw [← whole_host_preserved o.algebra c r s hm.shape (NativeProgram.shapeABI hm ha) hi v before]
    exact hh
  · apply copyback_precise r v before raw
    intro i hnot
    have hnw : ¬ (FPHostSource.effective c r s).writes r A layoutState
        (A.flat,A.addr r.out (Logical.coordOffset c i)) := by
      intro hw
      have hw' := Logical.writes_implies_written hm' i hw
      change Logical.Written (NativeIndex.capCall c (FPHostSource.metadataMaximum c r s)) i at hw'
      rw [FPHostSource.metadataMaximum_eq hi,← HostPlan.metadataMaximum_eq hm] at hw'
      exact hnot ((HostSource.written_effective hm i).mp hw')
    rw [hframe r.out (Logical.coordOffset c i) hnw]
    exact hinit i

end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPHostPrecise
