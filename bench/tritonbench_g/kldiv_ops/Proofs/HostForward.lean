import bench.tritonbench_g.kldiv_ops.Proofs.Launch
import bench.tritonbench_g.kldiv_ops.Proofs.Preparation

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000

 theorem bind_success {first : Action A} {next : A→Action B} {s t : State} {b : B}
    (h : bind first next s=.ok b t) : ∃ a u,first s=.ok a u ∧ next a u=.ok b t := by
  cases hf : first s with
  | error e failed => simp [bind,hf] at h
  | ok a u => exact ⟨a,u,rfl,by simpa [bind,hf] using h⟩

theorem shape2_success {a : Tensor} {s t : State} {dims : Nat×Nat}
    (h : shape2 a s=.ok dims t) : a.shape=[dims.1,dims.2] ∧ t=s := by
  unfold shape2 at h
  split at h
  next rows cols he => cases h; exact ⟨he,rfl⟩
  next => cases h

def finishForward (api : Services Word) (red : Reduction) (rows cols : Nat) (out : Tensor) : Action Tensor :=
  match red with
  | .none => pure out
  | .sum => api.sum out Bool.true
  | .mean => do
    let sum ← api.sum out Bool.false
    api.divNat sum (rows*cols)
  | .batchmean => do
    let sum ← api.sum out Bool.false
    api.divNat sum rows

def reducedWord (num : Numerics Word) (red : Reduction) (rows cols : Nat)
    (values : Nat→Word) (i : Nat) : Word :=
  match red with
  | .none => values i
  | .sum => num.sum Bool.true ((List.range rows).map values)
  | .mean => num.divide (num.sum Bool.false ((List.range rows).map values)) (rows*cols)
  | .batchmean => num.divide (num.sum Bool.false ((List.range rows).map values)) rows

structure Reduced {Word : Type} (codec : Codec Word) (num : Numerics Word) (red : Reduction)
    (rows cols : Nat) (out returned : Tensor) (before after : State) : Prop where
  shape : returned.shape=if red=.none then [rows,cols] else []
  live : returned.backing∈after.live
  preserved : Preserves before after
  identity : red=.none → returned=out
  values : ∀ i,i<returned.numel → value codec num.neg after.heap returned i=
    reducedWord num red rows cols (value codec num.neg before.heap out) i

theorem finishForward_correct {Word : Type} (codec : Codec Word) (num : Numerics Word)
    (api : Services Word) (lib : Library codec num api) (red : Reduction) (rows cols : Nat)
    (out returned : Tensor) (s t : State) (live : out.backing∈s.live)
    (shape : out.shape=red.outputShape rows cols)
    (call : finishForward api red rows cols out s=.ok returned t) :
    Reduced codec num red rows cols out returned s t := by
  cases red with
  | none =>
    simp only [finishForward,pure,Result.ok.injEq] at call
    rcases call with ⟨rfl,rfl⟩
    exact ⟨shape,live,Preserves.refl s,fun _=>rfl,fun _ _=>rfl⟩
  | sum =>
    have hsh : out.shape=[rows] := shape
    have h := lib.sum out Bool.true s returned t live (by simp [hsh]) call
    refine ⟨h.1.shape,h.1.live,h.1.preserved,by simp,?_⟩
    intro i hi
    have iz : i=0 := by simpa [Tensor.numel,h.1.shape] using hi
    subst i
    simpa [reducedWord,Tensor.numel,hsh] using h.2
  | mean =>
    obtain ⟨sum,u,hs,hd⟩ := bind_success call
    have hsh : out.shape=[rows] := shape
    have a := lib.sum out Bool.false s sum u live (by simp [hsh]) hs
    have b := lib.divide sum (rows*cols) u returned t a.1.live a.1.shape hd
    refine ⟨b.1.shape,b.1.live,a.1.preserved.trans b.1.preserved,by simp,?_⟩
    intro i hi
    have iz : i=0 := by simpa [Tensor.numel,b.1.shape] using hi
    subst i
    rw [b.2,a.2]
    simp [reducedWord,Tensor.numel,hsh]
  | batchmean =>
    obtain ⟨sum,u,hs,hd⟩ := bind_success call
    have hsh : out.shape=[rows] := shape
    have a := lib.sum out Bool.false s sum u live (by simp [hsh]) hs
    have b := lib.divide sum rows u returned t a.1.live a.1.shape hd
    refine ⟨b.1.shape,b.1.live,a.1.preserved.trans b.1.preserved,by simp,?_⟩
    intro i hi
    have iz : i=0 := by simpa [Tensor.numel,b.1.shape] using hi
    subst i
    rw [b.2,a.2]
    simp [reducedWord,Tensor.numel,hsh]

structure ForwardTrace {Word : Type} (api : Services Word) (source : SourceActions Word)
    (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry : State) (returned : Tensor) (final : State) where
  (rows cols : Nat)
  red : Reduction
  (out py pt : Tensor)
  (s0 s1 allocated packed ready launched : State)
  shape : shape2 prediction entry=.ok (rows,cols) s0
  configure : api.configure rows cols s0=.ok () s1
  reduction : lookup name s1=.ok red s1
  allocation : api.zeros (red.outputShape rows cols) prediction.device s1=.ok out allocated
  predictionCall : prepare api prediction allocated=.ok py packed
  targetCall : prepare api target packed=.ok pt ready
  capacity : rows*cols ≤ pt.numel
  launch : source.forward rows cols py pt out flag red eps ready=.ok () launched
  finishCall : finishForward api red rows cols out launched=.ok returned final

theorem lookup_state {name : String} {s t : State} {red : Reduction}
    (h : lookup name s=.ok red t) : t=s := by
  unfold lookup at h
  split at h <;> first | cases h; rfl | skip
  split at h <;> first | cases h; rfl | skip
  split at h <;> first | cases h; rfl | skip
  split at h <;> cases h <;> rfl

theorem forward_trace {Word : Type} (api : Services Word) (source : SourceActions Word)
    (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry : State) (returned : Tensor) (final : State)
    (call : forward api source prediction target flag name eps entry=.ok returned final) :
    Nonempty (ForwardTrace api source prediction target flag name eps entry returned final) := by
  obtain ⟨⟨rows,cols⟩,s0,hshape,h⟩ := bind_success call
  obtain ⟨⟨⟩,s1,hconfigure,h⟩ := bind_success h
  obtain ⟨red,s2,hlookup,h⟩ := bind_success h
  have eq := lookup_state hlookup
  subst s2
  obtain ⟨out,allocated,halloc,h⟩ := bind_success h
  obtain ⟨py,packed,hpy,h⟩ := bind_success h
  obtain ⟨pt,ready,hpt,h⟩ := bind_success h
  obtain ⟨⟨⟩,checked,hcap,h⟩ := bind_success h
  have cap := capacity_success hcap
  have eq := cap.2
  subst checked
  obtain ⟨⟨⟩,launched,hl,hf⟩ := bind_success h
  exact ⟨⟨rows,cols,red,out,py,pt,s0,s1,allocated,packed,ready,launched,
    hshape,hconfigure,hlookup,halloc,hpy,hpt,cap.1,hl,hf⟩⟩

structure ForwardResult {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (compiler : Compiler)
    (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry : State) (returned : Tensor) (final : State) where
  trace : ForwardTrace api (sourceActions codec M compiler) prediction target flag name eps entry returned final
  predictionShape : prediction.shape=[trace.rows,trace.cols]
  targetPackedShape : trace.pt.shape=target.shape
  allocation : Allocated (trace.red.outputShape trace.rows trace.cols) prediction.device trace.out entry trace.allocated
  predictionValues : ∀ i,i<prediction.numel → value codec num.neg trace.ready.heap trace.py i=
    value codec num.neg entry.heap prediction i
  targetValues : ∀ i,i<target.numel → value codec num.neg trace.ready.heap trace.pt i=
    value codec num.neg entry.heap target i
  launch : ForwardData codec M compiler trace.rows trace.cols trace.py trace.pt trace.out flag trace.red eps trace.ready trace.launched
  reduction : Reduced codec num trace.red trace.rows trace.cols trace.out returned trace.launched final
  callerFrame : Preserves entry final

theorem ForwardResult.capacity {Word : Type} [Inhabited Word] {codec : Codec Word} {num : Numerics Word}
    {M : Algebra Word} {api : Services Word} {compiler : Compiler}
    {prediction target : Tensor} {flag : Bool} {name : String} {eps : Word}
    {entry : State} {returned : Tensor} {final : State}
    (f : ForwardResult codec num M api compiler prediction target flag name eps entry returned final) :
    prediction.numel ≤ target.numel := by
  simpa [Tensor.numel,f.predictionShape,f.targetPackedShape] using f.trace.capacity

theorem forward_correct {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (lib : Library codec num api) (compiler : Compiler)
    (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry : State) (returned : Tensor) (final : State)
    (predLive : prediction.backing∈entry.live) (targetLive : target.backing∈entry.live)
    (call : forward api (sourceActions codec M compiler) prediction target flag name eps entry=.ok returned final) :
    Nonempty (ForwardResult codec num M api compiler prediction target flag name eps entry returned final) := by
  obtain ⟨tr⟩ := forward_trace api (sourceActions codec M compiler) prediction target flag name eps entry returned final call
  have sh := shape2_success tr.shape
  have conf := lib.configure tr.rows tr.cols tr.s0 tr.s1 tr.configure
  have start : tr.s1=entry := conf.trans sh.2
  have alloc := (lib.zeros _ _ _ _ _ tr.allocation).1
  have py := prepare_correct codec num api lib prediction tr.py tr.allocated tr.packed
    (alloc.preserved.live _ (by simpa only [start] using predLive)) tr.predictionCall
  have pt := prepare_correct codec num api lib target tr.pt tr.packed tr.ready
    (py.1.preserved.live _ (alloc.preserved.live _ (by simpa only [start] using targetLive))) tr.targetCall
  have before := alloc.preserved.trans (py.1.preserved.trans pt.1.preserved)
  have outLive := pt.1.preserved.live _ (py.1.preserved.live _ alloc.live)
  have source := forward_complete codec M compiler tr.rows tr.cols tr.py tr.pt tr.out flag tr.red eps tr.ready
  rw [show launchForward codec M compiler tr.rows tr.cols tr.py tr.pt tr.out flag tr.red eps tr.ready=
    .ok () tr.launched from tr.launch] at source
  obtain ⟨launched⟩ := source
  have returnedFacts := finishForward_correct codec num api lib tr.red tr.rows tr.cols tr.out returned
    tr.launched final (by rw [launched.live]; exact outLive) alloc.shape tr.finishCall
  have toLaunch : Preserves entry tr.launched := by
    refine ⟨?_,?_⟩
    · intro b hb
      rw [launched.live]
      exact before.live b (by simpa only [start] using hb)
    · intro b hb
      have bf : Frame entry.heap tr.ready.heap b := by
        simpa only [start] using before.frame b (by simpa only [start] using hb)
      exact bf.trans (launched.frame b (alloc.fresh b (by simpa only [start] using hb)))
  refine ⟨⟨tr,sh.1,pt.1.shape,
    by simpa only [start] using alloc,?_,?_,launched,returnedFacts,toLaunch.trans returnedFacts.preserved⟩⟩
  · intro i hi
    have hipy : i<tr.py.numel := by simpa only [Tensor.numel,py.1.shape] using hi
    rw [preserved_value codec num.neg pt.1.preserved tr.py py.1.live i hipy,py.1.values i hi]
    exact preserved_value codec num.neg (by simpa only [start] using alloc.preserved) prediction predLive i hi
  · intro i hi
    rw [pt.1.values i hi]
    exact preserved_value codec num.neg (by simpa only [start] using alloc.preserved.trans py.1.preserved) target targetLive i hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
