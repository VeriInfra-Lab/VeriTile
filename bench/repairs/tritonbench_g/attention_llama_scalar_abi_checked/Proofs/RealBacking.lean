import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.NativeBacking

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama LoopInvariant HostGrid HostResult Correctness
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def realConfig (c : Context) (Out : RegionName) (soz soh som son : Nat) (fp8 : Bool) : Source.Config :=
  ⟨c.Q,c.K,c.V,Out,c.sqz,c.sqh,c.sqm,c.sqk,c.skz,c.skh,c.skn,c.skk,
    c.svz,c.svh,c.svk,c.svn,soz,soh,som,son,c.NH,c.H,c.N,c.SP,c.D,c.causal,fp8⟩

def realContext (c : Context) (p : RegionName→Pointer) (before : Heap MemCell) : Context :=
  {c with initial:={c.initial with mem:=fun r offset=>before ((p r).address offset)}}

def realView (c : Context) (fp8 : Bool) : Context :=
  {c with initial:=if fp8 then FP8RealMemory.decoded c.K c.V c.initial else c.initial}

def realSource (c : Context) (Out : RegionName) (soz soh som son : Nat) (fp8 : Bool) : ComputeKernel :=
  if fp8 then ByteHostResult.kernel c Out soz soh som son else HostResult.kernel c Out soz soh som son

theorem context_owner (c : Context) (hD : 0<c.D) (i : Index B c.NH c.H c.D) :
    contextFor c (anchor hD (owner i))=contextFor c i := by
  have h:=owner_anchor hD (owner i)
  have h0:=congrArg (fun p=>p.1.val) h
  have h1:=congrArg (fun p=>p.2.val) h
  change (anchor hD (owner i)).query.val/64=i.query.val/64 at h0
  change (anchor hD (owner i)).batch.val*c.NH+(anchor hD (owner i)).head.val=
    i.batch.val*c.NH+i.head.val at h1
  simp only [contextFor,h0,h1]

theorem real_address (c : Context) (Out : RegionName) (soz soh som son : Nat) (fp8 : Bool)
    (i : Index B c.NH c.H c.D) :
    alOutOffset (contextFor c i).initial c.NH soz soh som son 64 (HostResult.lane i,i.channel,())=
      Tensor.offset (realConfig c Out soz soh som son fp8) i := by
  simpa only [outputAddress,Nat.zero_add,Tensor.offset,realConfig] using
    outputAddress_eq c 0 soz soh som son i

theorem real_lane_address (c : Context) (Out : RegionName) (soz soh som son : Nat) (fp8 : Bool)
    (hD : 0<c.D) (program : Program B c.NH c.H) (j : TileIndex [64,c.D])
    (hj : program.1.val*64+j.1.val<c.H) :
    alOutOffset (contextFor c (anchor hD program)).initial c.NH soz soh som son 64 j=
      Tensor.offset (realConfig c Out soz soh som son fp8)
        (laneIndex (realConfig c Out soz soh som son fp8) hD program j hj) := by
  have h0:=congrArg (fun p=>p.1.val) (owner_anchor hD program)
  change (anchor hD program).query.val/64=program.1.val at h0
  have hb:=MaskCoverage.batch_head_roundtrip c.NH (anchor hD program).batch.val
    (anchor hD program).head.val (anchor hD program).head.isLt
  simp only [alOutOffset,alBase,alRow,contextFor,Nat.one_ne_zero,↓reduceIte,h0,hb.1,hb.2,
    Tensor.offset,laneIndex,realConfig]

def realRead (cell : MemCell) : ℝ :=
  match cell.readAs .real with | some (some value)=>value | _=>0

def realPost (c : Context) (Out : RegionName) (soz soh som son : Nat) (fp8 : Bool)
    (o : Objects B (realConfig c Out soz soh som son fp8) p) (before : Heap MemCell)
    (states : Program B c.NH c.H→BlockState) : Heap MemCell :=
  merge before (tensorAddress o.output) owner
    (fun program i=>(states program).mem Out (Tensor.offset (realConfig c Out soz soh som son fp8) i))

theorem real_block (c : Context) (Out : RegionName) (soz soh som son : Nat) (fp8 : Bool)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (hu : ∀ r offset,c.initial.undef r offset=0)
    (layout : OutputLayout B c)
    (hs : layout.soz=soz ∧ layout.soh=soh ∧ layout.som=som ∧ layout.son=son)
    (domain : fp8=Bool.true → ByteHostResult.Domain (B:=B) (heads:=c.NH) c)
    (sep : fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ Out≠c.K ∧ Out≠c.V)
    (i : Index B c.NH c.H c.D) :
    ∃ sf, exec (realSource c Out soz soh som son fp8).eraseDType.toAlgKernel
        (contextFor c i).initial=some sf ∧
      (∀ j : TileIndex [64,c.D], alActive (contextFor c i).initial c.H 64 j →
        sf.readMem Out (alOutOffset (contextFor c i).initial c.NH soz soh som son 64 j)=
          RealKernel.target (realView (contextFor c i) fp8) j) ∧
      (∀ r offset, (r≠Out ∨ ∀ j : TileIndex [64,c.D], alActive (contextFor c i).initial c.H 64 j →
          offset≠alOutOffset (contextFor c i).initial c.NH soz soh som son 64 j) →
        sf.mem r offset=c.initial.mem r offset) := by
  have hout : Postamble.ActiveInjective ((contextFor c i).initial.pids 0*64) c.H 64 c.D som son := by
    simpa only [hs.2.2.1,hs.2.2.2] using layout.blockInjective i
  have hsp : c.causal=Bool.true → c.SP≤(contextFor c i).initial.pids 0*64 := by
    intro hc; rw [hSP hc]; omega
  cases fp8
  · exact RealKernel.execution (contextFor c i) Out soz soh som son hD hN hsp hout hu
  · have hd:=domain rfl
    have hs':=sep rfl
    exact FP8Kernel.execution (contextFor c i) Out soz soh som son hD hN hsp hout
      hs'.1 hs'.2.1 hs'.2.2.1 hs'.2.2.2 hu (hd.keys i) (hd.values i)

/-- Mathematical interpretation of the same source stores in one actual
caller backing. FP16 and E5M2 share this theorem; the byte view decodes K/V only.
No native-to-Real agreement is asserted. -/
theorem real_backing (raw : RawScalar) (c : Context) (Out : RegionName)
    (soz soh som son : Nat) (fp8 : Bool)
    (_scale : ScalarABI.realValue raw=some c.scale)
    (hD : 0<c.D) (hN : 0<c.N) (hSP : c.causal=Bool.true → c.SP=0)
    (hu : ∀ r offset,c.initial.undef r offset=0)
    (p : RegionName→Pointer) (o : Objects B (realConfig c Out soz soh som son fp8) p)
    (before : Heap MemCell)
    (x : HostResult.Inputs B c.NH c.H c.D)
    (binding : HostResult.Binding (realView (realContext c p before) fp8) x)
    (domain : fp8=Bool.true → ByteHostResult.Domain (B:=B) (heads:=c.NH) (realContext c p before))
    (sep : fp8=Bool.true → c.Q≠c.K ∧ c.Q≠c.V ∧ Out≠c.K ∧ Out≠c.V)
    (hinj : Function.Injective (Tensor.offset (B:=B) (realConfig c Out soz soh som son fp8))) :
    ∃ states : Program B c.NH c.H→BlockState,
      (∀ program, exec (realSource c Out soz soh som son fp8).eraseDType.toAlgKernel
        (contextFor (realContext c p before) (anchor hD program)).initial=some (states program)) ∧
      (∀ program region offset, (region≠Out ∨ ∀ j : TileIndex [64,c.D],
        alActive (contextFor (realContext c p before) (anchor hD program)).initial c.H 64 j →
          offset≠alOutOffset (contextFor (realContext c p before) (anchor hD program)).initial
            c.NH soz soh som son 64 j) →
        (states program).mem region offset=before ((p region).address offset)) ∧
      let returned : Returned MemCell := ⟨o.output,realPost c Out soz soh som son fp8 o before states⟩
      (∀ i : Index B c.NH c.H c.D, realRead (returned.read i)=HostResult.expected c x i) ∧
      (∀ location, (∀ i : Index B c.NH c.H c.D,location≠tensorAddress o.output i) →
        returned.backing location=before location) ∧
      (∀ t∈[o.query,o.key,o.value], ∀ location,
        t.allocationBase≤location → location<t.allocationBase+t.allocationBytes →
          returned.backing location=before location) ∧
      (∀ program (j : TileIndex [64,c.D]), alActive (contextFor (realContext c p before) (anchor hD program)).initial c.H 64 j →
        ∃ i : Index B c.NH c.H c.D, owner i=program ∧
          alOutOffset (contextFor (realContext c p before) (anchor hD program)).initial c.NH soz soh som son 64 j=
            Tensor.offset (realConfig c Out soz soh som son fp8) i) ∧
      (∀ i j : Index B c.NH c.H c.D,tensorAddress o.output i=tensorAddress o.output j → owner i=owner j) := by
  let cc:=realContext c p before
  let nc:=realConfig c Out soz soh som son fp8
  let layout : OutputLayout B cc := {
    base:=0,soz:=soz,soh:=soh,som:=som,son:=son
    injective:=by
      intro i j he
      apply hinj
      simpa only [outputAddress_eq,Nat.zero_add,Tensor.offset,nc,realConfig,cc,realContext] using he }
  have runs:=fun program : Program B c.NH c.H =>
    real_block cc Out soz soh som son fp8 hD hN hSP hu layout ⟨rfl,rfl,rfl,rfl⟩ domain sep (anchor hD program)
  choose states he hv hf using runs
  refine ⟨states,he,hf,?_,?_,?_,?_,?_⟩
  · intro i
    change realRead (realPost c Out soz soh som son fp8 o before states (tensorAddress o.output i))=_
    rw [realPost]
    have hm:=merge_read before (fun j : Index B c.NH c.H c.D=>tensorAddress o.output j) owner
      (fun program j=>(states program).mem Out (Tensor.offset nc j))
      (output_injective nc p o hinj) i
    change realRead (merge before (fun j : Index B c.NH c.H c.D=>tensorAddress o.output j) owner
      (fun program j=>(states program).mem Out (Tensor.offset nc j)) (tensorAddress o.output i))=_
    rw [hm]
    have hr : alActive (contextFor cc i).initial c.H 64 (HostResult.lane i,i.channel,()) := by
      change i.query.val/64*64+i.query.val%64<c.H
      have hi:=i.query.isLt
      omega
    have vi:=hv (owner i) (HostResult.lane i,i.channel,())
    have vi' : alActive (contextFor cc i).initial c.H 64 (HostResult.lane i,i.channel,()) →
        (states (owner i)).readMem Out
          (alOutOffset (contextFor cc i).initial c.NH soz soh som son 64 (HostResult.lane i,i.channel,()))=
        RealKernel.target (realView (contextFor cc i) fp8) (HostResult.lane i,i.channel,()) := by
      let sa:=(contextFor cc (anchor hD (owner i))).initial
      let si:=(contextFor cc i).initial
      have hs : sa=si := congrArg Context.initial (context_owner cc hD i)
      change alActive sa c.H 64 (HostResult.lane i,i.channel,()) →
        (states (owner i)).readMem Out (alOutOffset sa c.NH soz soh som son 64 (HostResult.lane i,i.channel,()))=
          RealKernel.target (realView {cc with initial:=sa} fp8) (HostResult.lane i,i.channel,()) at vi
      rw [hs] at vi
      exact vi
    have ha:=real_address cc Out soz soh som son fp8 i
    have va : realRead ((states (owner i)).mem Out (Tensor.offset nc i))=
        RealKernel.target (realView (contextFor cc i) fp8) (HostResult.lane i,i.channel,()) :=
      (congrArg ((states (owner i)).readMem Out) ha).symm.trans (vi' hr)
    apply va.trans
    have target:=HostResult.target_binding (realView cc fp8) x binding i
    cases fp8 <;> simpa only [realView,FP8Kernel.view,contextFor,cc,realContext,
      FP8RealMemory.decoded,HostResult.expected,↓reduceIte] using target
  · intro location miss
    change merge before (tensorAddress o.output) owner
      (fun program i=>(states program).mem Out (Tensor.offset nc i)) location=before location
    exact merge_frame _ _ _ _ location miss
  · intro t ht location hlo hhi
    change merge before (tensorAddress o.output) owner
      (fun program i=>(states program).mem Out (Tensor.offset nc i)) location=before location
    apply merge_frame
    intro i
    have ho:=layout_bounds o.output (p Out) _ _ o.outputLayout i
    have hfr:=o.fresh t ht
    dsimp only [nc,realConfig] at *
    omega
  · intro program j hj
    have h0:=congrArg (fun q=>q.1.val) (owner_anchor hD program)
    change (anchor hD program).query.val/64=program.1.val at h0
    have hj' : program.1.val*64+j.1.val<c.H := by
      change (anchor hD program).query.val/64*64+j.1.val<c.H at hj
      rw [h0] at hj
      exact hj
    exact ⟨laneIndex nc hD program j hj',owner_lane nc hD program j hj',
      real_lane_address cc Out soz soh som son fp8 hD program j hj'⟩
  · intro i j hij
    exact congrArg owner (output_injective nc p o hinj hij)

#axiomsClean real_block
#axiomsClean real_backing
#specNonCircular HostResult.expected avoiding [realPost,merge,Triton.exec]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Backing
