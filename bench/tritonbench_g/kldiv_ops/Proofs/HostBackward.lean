import bench.tritonbench_g.kldiv_ops.Proofs.HostBackwardFinish

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000

structure BackwardTrace {Word : Type} (api : Services Word) (source : SourceActions Word)
    (target grad dst : Tensor) (flag : Bool) (entry : State) (returned : Tensor) (final : State) where
  (rows cols : Nat)
  (pt pg work : Tensor)
  (s0 s1 packed cloned ready launched : State)
  shape : shape2 target entry=.ok (rows,cols) s0
  configure : api.configure rows cols s0=.ok () s1
  targetCall : prepare api target s1=.ok pt packed
  gradientCall : snapshot api grad packed=.ok pg cloned
  allocation : api.emptyContiguousLike dst cloned=.ok work ready
  capacity : rows*cols ≤ work.numel
  launch : source.backward rows cols pt work flag ready=.ok () launched
  finishCall : backwardFinish api pg dst work launched=.ok returned final

theorem backward_trace {Word : Type} (api : Services Word) (source : SourceActions Word)
    (target grad dst : Tensor) (flag : Bool) (entry : State) (returned : Tensor) (final : State)
    (call : backward api source target grad dst flag entry=.ok returned final) :
    Nonempty (BackwardTrace api source target grad dst flag entry returned final) := by
  obtain ⟨⟨rows,cols⟩,s0,hshape,h⟩ := bind_success call
  obtain ⟨⟨⟩,s1,hconfigure,h⟩ := bind_success h
  obtain ⟨pt,packed,hpt,h⟩ := bind_success h
  obtain ⟨pg,cloned,hpg,h⟩ := bind_success h
  obtain ⟨work,ready,hw,h⟩ := bind_success h
  obtain ⟨⟨⟩,checked,hcap,h⟩ := bind_success h
  have cap := capacity_success hcap
  have eq := cap.2
  subst checked
  obtain ⟨⟨⟩,launched,hl,hf⟩ := bind_success h
  exact ⟨⟨rows,cols,pt,pg,work,s0,s1,packed,cloned,ready,launched,hshape,hconfigure,hpt,hpg,hw,cap.1,hl,hf⟩⟩

structure BackwardResult {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (compiler : Compiler)
    (target grad dst : Tensor) (flag : Bool) (entry : State) (returned : Tensor) (final : State) where
  trace : BackwardTrace api (sourceActions codec M compiler) target grad dst flag entry returned final
  targetShape : target.shape=[trace.rows,trace.cols]
  gradientShape : trace.pg.shape=grad.shape
  workShape : trace.work.shape=dst.shape
  targetValues : ∀ i,i<target.numel → value codec num.neg trace.ready.heap trace.pt i=value codec num.neg entry.heap target i
  gradientValues : ∀ i,i<grad.numel → value codec num.neg trace.launched.heap trace.pg i=value codec num.neg entry.heap grad i
  launch : BackwardData codec M compiler trace.rows trace.cols trace.pt trace.work flag trace.ready trace.launched
  finish : BackwardFinished codec num trace.pg dst trace.work returned trace.launched final
  callerFrame : Modified dst entry final

theorem BackwardResult.capacity {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
    {M : Algebra Word} {api : Services Word} {compiler : Compiler}
    {target grad dst : Tensor} {flag : Bool} {entry : State} {returned : Tensor} {final : State}
    (f : BackwardResult codec num M api compiler target grad dst flag entry returned final) :
    target.numel ≤ dst.numel := by
  simpa [Tensor.numel,f.targetShape,f.workShape] using f.trace.capacity

theorem backward_correct {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (lib : Library codec num api) (compiler : Compiler)
    (target grad dst : Tensor) (flag : Bool) (entry : State) (returned : Tensor) (final : State)
    (tlive : target.backing∈entry.live) (glive : grad.backing∈entry.live) (dlive : dst.backing∈entry.live)
    (call : backward api (sourceActions codec M compiler) target grad dst flag entry=.ok returned final) :
    Nonempty (BackwardResult codec num M api compiler target grad dst flag entry returned final) := by
  obtain ⟨tr⟩ := backward_trace api (sourceActions codec M compiler) target grad dst flag entry returned final call
  have sh := shape2_success tr.shape
  have conf := lib.configure tr.rows tr.cols tr.s0 tr.s1 tr.configure
  have start : tr.s1=entry := conf.trans sh.2
  have pt := prepare_correct codec num api lib target tr.pt tr.s1 tr.packed
    (by simpa only [start] using tlive) tr.targetCall
  have pg := snapshot_correct codec num api lib grad tr.pg tr.packed tr.cloned
    (pt.1.preserved.live _ (by simpa only [start] using glive)) tr.gradientCall
  have entryCloned : Preserves entry tr.cloned := by simpa only [start] using pt.1.preserved.trans pg.1.preserved
  have alloc := lib.empty dst tr.cloned tr.work tr.ready (entryCloned.live _ dlive) tr.allocation
  have entryReady := entryCloned.trans alloc.preserved
  have gradReady := alloc.preserved.live _ pg.1.live
  have source := backward_complete codec M compiler tr.rows tr.cols tr.pt tr.work flag tr.ready
  rw [show launchBackward codec M compiler tr.rows tr.cols tr.pt tr.work flag tr.ready=
    .ok () tr.launched from tr.launch] at source
  obtain ⟨launched⟩ := source
  have apartGrad : tr.work.backing.disjoint tr.pg.backing := alloc.fresh _ pg.1.live
  have dstGrad : dst.backing.disjoint tr.pg.backing := Backing.disjoint_symm _ _
    (pg.2.2 dst.backing (pt.1.preserved.live _ (by simpa only [start] using dlive)))
  obtain ⟨returnedFacts⟩ := backwardFinish_correct codec num api lib tr.pg dst tr.work returned tr.launched final
    (by rw [launched.live]; exact gradReady)
    (by rw [launched.live]; exact entryReady.live _ dlive)
    (by rw [launched.live]; exact alloc.live) alloc.shape.symm dstGrad tr.finishCall
  have entryLaunch : Preserves entry tr.launched := by
    refine ⟨?_,?_⟩
    · intro b hb
      rw [launched.live]
      exact entryReady.live b hb
    · intro b hb
      exact (entryReady.frame b hb).trans (launched.frame b (alloc.fresh b (entryCloned.live b hb)))
  refine ⟨⟨tr,sh.1,pg.1.shape,alloc.shape,?_,?_,launched,returnedFacts,
    (entryLaunch.modified dst).trans returnedFacts.callerFrame⟩⟩
  · intro i hi
    have hit : i<tr.pt.numel := by simpa only [Tensor.numel,pt.1.shape] using hi
    rw [preserved_value codec num.neg (pg.1.preserved.trans alloc.preserved) tr.pt pt.1.live i hit,pt.1.values i hi]
    simp only [start]
  · intro i hi
    have hig : i<tr.pg.numel := by simpa only [Tensor.numel,pg.1.shape] using hi
    rw [Storage.preserves_value codec num.neg tr.ready.heap tr.launched.heap tr.pg
      (launched.frame _ apartGrad) i hig]
    rw [preserved_value codec num.neg alloc.preserved tr.pg pg.1.live i hig,pg.1.values i hi]
    exact preserved_value codec num.neg (by simpa only [start] using pt.1.preserved) grad glive i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
