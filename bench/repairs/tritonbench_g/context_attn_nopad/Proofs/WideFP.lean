import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideNative
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideLogical
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPHostPrecise

/-! Complete opaque FP success and precise caller/flat-memory frame on the
wider metadata domain. The full source prelude/body/range/epilogue and precise
store theorem are reused unchanged. No FP output or successful run is assumed. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.WideFP
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula Host HostPlan FPHostSource
noncomputable section
set_option maxHeartbeats 2000000
variable {types : WideNative.MetadataTypes}

theorem atProgram_eq {α : Type} {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} {j : Job c} (hp : WideNative.Projects c r A layoutState types j) (s : State α) :
    atProgram c j s = FPHost.atProgram c (index c j) s := by
  unfold atProgram FPHost.atProgram
  rw [WideNative.nativePids_eq hp]

theorem launch_eq {α : Type} {c : Call} {r : Regions} {A : FlatAlloc}
    {layoutState : BlockState} {j : Job c} (hp : WideNative.Projects c r A layoutState types j) (s : State α) :
    FPSymbolic.launchOf (atProgram c j s) = c.launch (index c j) := by
  rw [atProgram_eq hp]
  simp [FPSymbolic.launchOf,FPHost.atProgram,Call.launch,GridIndex.toPids,Call.grid,Grid.rank]

/-- Reuse the already checked source write/mask formulas and original flat
footprint. This connects an opaque-FP store to the independent host footprint. -/
theorem writes_subset {α : Type} [Inhabited α]
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State α}
    (hm : WideDomain.Memory c r A layoutState) (hi : Metadata c r s)
    (j : Job c) (hp : WideNative.Projects c r A layoutState types j) (rg : RegionName) (off : Nat)
    (hw : FPSymbolic.Writes c.layout r (atProgram c j s) rg off) :
    c.writes r A layoutState (A.flat,A.addr rg off) := by
  obtain ⟨lane,hactive,hrg,hoff⟩ := hw
  rw [launch_eq hp s] at hactive hoff
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
    (hm : WideDomain.Memory c r A layoutState) (hi : Metadata c r s)
    (j : Job c) (hp : WideNative.Projects c r A layoutState types j) :
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
    exact hframe rg off (fun hw => hnot (writes_subset hm hi j hp rg off hw))

/-- Successful complete native stream with the exact host write footprint,
including cells in the output region. Metadata stability is proved at each step. -/
theorem programs_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : WideDomain.Memory c r A layoutState)
    (hp : ∀ j : Job c, WideNative.Projects c r A layoutState types j) (order : List (Job c)) (s : State α) (hi : Metadata c r s) :
    ∃ t, runPrograms o.algebra c r order s = some t ∧ Metadata c r t ∧
      (∀ rg off, ¬ c.writes r A layoutState (A.flat,A.addr rg off) → t.mem rg off = s.mem rg off) := by
  induction order generalizing s with
  | nil => exact ⟨s,rfl,hi,by intros; rfl⟩
  | cons j rest ih =>
    obtain ⟨mid,he,hmeta,hframe⟩ := program_precise o hm hi j (hp j)
    obtain ⟨t,ht,hmetat,hframet⟩ := ih mid hmeta
    refine ⟨t,?_,hmetat,?_⟩
    · simp only [runPrograms,he,Option.bind_some,ht]
    · intro rg off hnot
      exact (hframet rg off hnot).trans (hframe rg off hnot)


theorem programs_project {α : Type} [Inhabited α] (M : Algebra α) {c : Call} {r : Regions}
    {A : FlatAlloc} {layoutState : BlockState}
    (hp : ∀ j : Job c, WideNative.Projects c r A layoutState types j)
    (order : List (Job c)) (s : State α) :
    runPrograms M c r order s = FPHost.runJobs M c r Bool.false (order.map (index c)) s := by
  induction order generalizing s with
  | nil => rfl
  | cons j rest ih =>
    simp only [runPrograms,List.map_cons,FPHost.runJobs,atProgram_eq (hp j)]
    change (FP.Structural.exec _ _ _).bind _ = (FP.Structural.exec _ _ _).bind _
    congr 1
    funext next
    exact ih next

theorem effective_covered {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (b t : Nat) (hb : b < c.batches) (ht : t < c.length b) :
    t < (HostPlan.effective c r s).blocks*128 ↔ t < c.blocks*128 := by
  change t < ((NativeIndex.capMaxInputLen c.maxInputLen (HostPlan.metadataMaximum c r s)+127)/128)*128 ↔ _
  rw [WideDomain.metadataMaximum_eq hm]
  exact NativeIndex.cap_preserves_active c.maxInputLen (maximum c) t
    (lt_of_lt_of_le ht (length_le_maximum c b hb))

theorem written_effective {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (i : Logical.Coord c) :
    Logical.Written (HostPlan.effective c r s) i ↔ Logical.Written c i := by
  constructor
  · rintro ⟨b,t,hb,ht,hg,hr,hh,hd⟩
    exact ⟨b,t,hb,ht,(effective_covered hm b t hb ht).mp hg,hr,hh,hd⟩
  · rintro ⟨b,t,hb,ht,hg,hr,hh,hd⟩
    exact ⟨b,t,hb,ht,(effective_covered hm b t hb ht).mpr hg,hr,hh,hd⟩

theorem whole_host_precise {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState} {s : State α}
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (fits : WideNative.MetadataFits c types) (hi : Metadata c r s)
    (v : OutputView α c) (before : Nat → α) (hinit : FPHostPrecise.Initialized c r s v before) :
    FPHostPrecise.Contract o.algebra c r A layoutState s v before := by
  have he : FPHostSource.effective c r s = HostPlan.effective c r layoutState := by
    unfold FPHostSource.effective HostPlan.effective
    rw [FPHostSource.metadataMaximum_eq hi,WideDomain.metadataMaximum_eq hm]
  have hm' : WideDomain.Memory (FPHostSource.effective c r s) r A layoutState := by
    exact WideNative.capMemory hm _
  have hp : ∀ j : Job (FPHostSource.effective c r s),
      WideNative.Projects (FPHostSource.effective c r s) r A layoutState types j := by
    rw [he]
    exact fun j => (WideNative.complete_projection hm ha types fits j).1
  have hi' : Metadata (FPHostSource.effective c r s) r s := ⟨hi.start,hi.length⟩
  obtain ⟨raw,hr,_,hframe⟩ := programs_precise o hm' hp (HostPlan.jobs _) s hi'
  have hh : runHost o.algebra c r s v before = some (finish r v before raw) := by
    dsimp only [runHost]
    rw [hr]
    rfl
  have href : reference o.algebra c r s v before = some (finish r v before raw) := by
    dsimp only [reference]
    rw [← programs_project o.algebra hp,hr]
    rfl
  refine ⟨raw,finish r v before raw,hr,hh,href,rfl,rfl,rfl,rfl,caller_readback r v before raw,?_,hframe⟩
  apply FPHostPrecise.copyback_precise r v before raw
  intro i hnot
  have hnw : ¬ (FPHostSource.effective c r s).writes r A layoutState
      (A.flat,A.addr r.out (Logical.coordOffset c i)) := by
    intro hw
    have hw' := WideLogical.writes_implies_written hm' i hw
    change Logical.Written (NativeIndex.capCall c (FPHostSource.metadataMaximum c r s)) i at hw'
    rw [FPHostSource.metadataMaximum_eq hi,← WideDomain.metadataMaximum_eq hm] at hw'
    exact hnot ((written_effective hm i).mp hw')
  rw [hframe r.out (Logical.coordOffset c i) hnw]
  exact hinit i

#print axioms whole_host_precise
end
end VeriTile.Bench.Repairs.ContextAttnNopad.WideFP
