import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.NativeBounds

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Returned
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
open Bytes
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _

structure Context where
  (suppliedOutput ownerDevice callerDevice : Nat)
structure Result where
  graph : State Native.Expr
  bytes : Memory
  returned : Option Unit
  (suppliedOutput currentDevice : Nat)
  launchDevice : Option Nat

def invoke (c : Call) (B : Windows extent) (profile : Profile) (ops : Native.WordISA)
    (ctx : Context) (before : Memory) : Option Result := do
  let graph ← FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)
  return ⟨graph,stores c B profile ops graph before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩

def WholeBytes (c : Call) (A : FlatAlloc) (B : Windows A.extent)
    (profile : Profile) (ops : Native.WordISA) (ctx : Context) (before : Memory) : Prop :=
  ∃ result,invoke c B profile ops ctx before=some result ∧
    result.returned=none ∧ result.suppliedOutput=ctx.suppliedOutput ∧ result.currentDevice=ctx.callerDevice ∧
    result.launchDevice=(if Needed c then some ctx.ownerDevice else none) ∧
    FPSuccess.runGrid c Native.scalar Native.dots (order c) (initial c B before)=some result.graph ∧
    result.graph.mem=(reference c B before).mem ∧
    (∀ i : Lane c,active c i → result.bytes (address c B i)=byte (referenceWord c B profile ops before i) i.2.2) ∧
    (∀ a,(¬∃ i : Lane c,active c i ∧ address c B i=a) → result.bytes a=before a) ∧
    (∀ rg∈Checked.readRegions c.regions,∀ off,off < A.extent rg → ∀ j : Fin 4,
      result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
    (∀ rg off,¬FPSuccess.writes c (order c) rg off → result.graph.mem rg off=(initial c B before).mem rg off) ∧
    (¬Needed c → result.bytes=before)

theorem whole_bytes (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions)
    (profile : Profile) (ops : Native.WordISA) (ctx : Context) : WholeBytes c A B profile ops ctx before := by
  change c.input=RealSetup.inputs (layoutState B before) c.regions at observed
  have guard := (Checked.checkInputs_iff c.args c.regions v A.extent c.input requestedMax).mp checked
  have hv := Checked.valid c.args c.regions A (layoutState B before) guard.2.1
    (by simpa only [observed] using guard.2.2.1) labels
  have metadata := Bytes.metadata c B before observed
  obtain ⟨graph,run,referenceEq⟩ := FPReference.whole c Native.scalar Native.dots (order c) (initial c B before)
  obtain ⟨framed,frameRun,_,framing⟩ := FPSuccess.grid_success_frame c Native.scalar Native.dots
    labels (order c) (initial c B before) metadata
  have eq : framed=graph := Option.some.inj (frameRun.symm.trans run)
  subst framed
  have active_needed (i : Lane c) (hi : active c i) : Needed c :=
    ⟨(Host.launchOf c.args i.1).batch,Host.batch_lt c.args i.1,lt_of_le_of_lt (by positivity) hi.1⟩
  refine ⟨⟨graph,stores c B profile ops graph before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩,?_,rfl,rfl,rfl,rfl,run,referenceEq,?_,?_,?_,framing,?_⟩
  · simp only [invoke,run]; rfl
  · intro i hi
    change stores c B profile ops graph before (address c B i)=_
    rw [readback c B (layoutState B before) observed.symm hv profile ops graph before i hi]
    unfold outputWord referenceWord
    rw [referenceEq]
    rfl
  · exact frame c B profile ops graph before
  · intro rg read off bound j
    by_cases needed : Needed c
    · exact input_frame c A (layoutState B before) B observed.symm hv
        (Checked.physical_separate c v B guard.1 guard.2.1 guard.2.2.1 storage needed)
        profile ops graph before rg read off bound j
    · apply frame
      rintro ⟨i,hi,_⟩
      exact needed (active_needed i hi)
  · intro empty
    funext a
    apply frame
    rintro ⟨i,hi,_⟩
    exact empty (active_needed i hi)


structure MachineFacts (c : Call) (v : Checked.Views) (A : FlatAlloc)
    (B : Windows A.extent) (before : Memory) : Prop where
  source : Host.Admissible c.args c.regions A (layoutState B before)
  metadata : FPSuccess.Metadata c (initial c B before)
  dotWidths : (Host.layout c.args).D%8=0 ∧ (Host.layout c.args).BN%8=0
  strides : v.q.strides=[(Host.layout c.args).qRow,(Host.layout c.args).qHead,(Host.layout c.args).qDim] ∧
    v.k.strides=[(Host.layout c.args).kRow,(Host.layout c.args).kHead,(Host.layout c.args).kDim] ∧
    v.v.strides=[(Host.layout c.args).vRow,(Host.layout c.args).vHead,(Host.layout c.args).vDim] ∧
    v.o.strides=[(Host.layout c.args).outRow,(Host.layout c.args).outHead,(Host.layout c.args).outDim] ∧
    v.pages.strides=[(Host.layout c.args).requestRow,(Host.layout c.args).requestToken]
  alignment : ∀ rg x,(rg,x)∈Checked.roleViews c.regions v → 0 < x.metadata.numel → ∀ off,(B.base rg+4*off)%4=0
  query : ∀ (idx : GridIndex (Host.grid c.args)) (i : Fin (Host.layout c.args).BM) (d : Fin (Host.layout c.args).D),
    let row := c.input.start (Host.launchOf c.args idx).batch+queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.val
    let head := (Host.launchOf c.args idx).head
    ∀ desc∈[c.args.q,c.args.o],
      (Checked.coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=desc.address row head d.val ∧
      (Checked.word64 4*Checked.coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=4*desc.address row head d.val
  cache : ∀ (idx : GridIndex (Host.grid c.args)) (step j : Nat) (d : Fin (Host.layout c.args).D),
    let row := Checked.gathered c idx step j
    let head := (Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup
    ∀ desc∈[c.args.k,c.args.v],
      (Checked.coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=desc.address row head d.val ∧
      (Checked.word64 4*Checked.coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=4*desc.address row head d.val
  pages : ∀ (idx : GridIndex (Host.grid c.args)) (step : Nat) (j : Fin (Host.layout c.args).BN),
    step < count (Host.layout c.args) (Host.launchOf c.args idx) c.input →
    let off := AddressBounds.tokenAddress (Host.layout c.args) (Host.launchOf c.args idx) c.input (step*(Host.layout c.args).BN+j.val)
    (Checked.word64 off).toInt=off ∧ (Checked.word64 4*Checked.word64 off).toInt=4*off
  masks : ∀ (idx : GridIndex (Host.grid c.args)) (i : Fin (Host.layout c.args).BM) (d : Fin (Host.layout c.args).D),
    let l := Host.launchOf c.args idx
    let row := queryRow (Host.layout c.args) l i.val
    ((Checked.word64 row).slt (Checked.word64 (c.input.sequence l.batch)-Checked.word64 (c.input.prompt l.batch))=
      decide ((row : Int)<c.input.sequence l.batch-c.input.prompt l.batch)) ∧
    ((Checked.word64 d.val).slt (Checked.word64 c.args.q.dims)=decide (d.val<c.args.q.dims)) ∧
    ∀ column : Nat,(column : Int) ≤ c.input.sequence l.batch+128 →
      (Checked.word64 column).sle (Checked.word64 row+Checked.word64 (c.input.prompt l.batch))=
        decide ((column : Int) ≤ row+c.input.prompt l.batch)
  groupedHead : ∀ idx : GridIndex (Host.grid c.args),
    ((Checked.word64 (Host.launchOf c.args idx).head).sdiv (Checked.word64 (Host.layout c.args).kvGroup)).toInt=
      ((Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup : Nat)
  pointer : ∀ rg off,off < A.extent rg → ∀ j : Fin 4,
    (BitVec.ofNat 64 (B.base rg)+BitVec.ofNat 64 4*BitVec.ofNat 64 off+BitVec.ofNat 64 j.val).toNat=B.base rg+4*off+j.val
  control : ∀ idx : GridIndex (Host.grid c.args),
    let l := Host.launchOf c.args idx
    let p := Host.layout c.args
    let upper := ((l.block+1)*p.BM : Nat)+c.input.prompt l.batch
    let sequence := c.input.sequence l.batch
    ((if (Checked.word64 upper).slt (Checked.word64 sequence) then Checked.word64 upper else Checked.word64 sequence).toInt=min upper sequence) ∧
    ∀ cur : Nat,(cur : Int) ≤ sequence+p.BN → (Checked.word64 cur+Checked.word64 p.BN).toInt=(cur+p.BN : Nat)

theorem machine_facts (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions) :
    MachineFacts c v A B before := by
  change c.input=RealSetup.inputs (layoutState B before) c.regions at observed
  have guard := (Checked.checkInputs_iff c.args c.regions v A.extent c.input requestedMax).mp checked
  exact ⟨Checked.valid c.args c.regions A (layoutState B before) guard.2.1
      (by simpa only [observed] using guard.2.2.1) labels,
    Bytes.metadata c B before observed,
    ⟨Checked.padded_eight c.args guard.2.1.2.2.2.2.2.2.2.2.2.2.2.2.1,Checked.block_eight c.args⟩,
    Checked.source_strides c.args c.regions v A.extent guard.1 guard.2.1,
    Checked.role_alignment c.args c.regions v B guard.1 storage,
    Checked.query_formed c v A.extent requestedMax guard.2.1 guard.2.2.1 guard.2.2.2,
    Checked.cache_formed c v A.extent requestedMax guard.2.1 guard.2.2.1 guard.2.2.2,
    Checked.page_formed c v A.extent requestedMax guard.2.1 guard.2.2.1 guard.2.2.2,
    Checked.source_masks c v A.extent requestedMax guard.2.1 guard.2.2.1 guard.2.2.2 B before observed,
    Checked.grouped_head c v requestedMax guard.2.2.2,
    Checked.pointer_exact B,
    Checked.direct_control c v requestedMax guard.2.2.1 guard.2.2.2 B before observed⟩

/-- Nine all-word instruction-local laws. No dot tile, reduction, source
execution, output, or returned Tensor equality is a field. -/
structure LocalLaws (a b : Native.WordISA) : Prop where
  literal : ∀ x,a.literal32 x=b.literal32 x
  add : ∀ x y,a.addRN x y=b.addRN x y
  sub : ∀ x y,a.subRN x y=b.subRN x y
  mul : ∀ x y,a.mulRN x y=b.mulRN x y
  div : ∀ x y,a.divFull x y=b.divFull x y
  max : ∀ x y,a.maxNum x y=b.maxNum x y
  exp : ∀ x,a.ex2Approx x=b.ex2Approx x
  fma : ∀ x y z,a.fmaRN x y z=b.fmaRN x y z
  mma : ∀ r c x y z,a.mmaTF32 r c x y z=b.mmaTF32 r c x y z

theorem LocalLaws.operations_eq (a b : Native.WordISA) (laws : LocalLaws a b) : a=b := by
  cases a
  cases b
  cases laws
  congr <;> funext <;> aesop

theorem whole_if_local (c : Call) (v : Checked.Views) (A : FlatAlloc) (B : Windows A.extent)
    (requestedMax : Nat) (checked : Checked.checkInputs c.args c.regions v A.extent c.input requestedMax=Bool.true)
    (before : Memory) (observed : Observed c B before)
    (storage : Checked.StorageBinding c.regions v B) (labels : Checked.Labels c.regions)
    (profile : Profile) (actual refOps : Native.WordISA) (laws : LocalLaws actual refOps) (ctx : Context) :
    WholeBytes c A B profile actual ctx before ∧ WholeBytes c A B profile refOps ctx before ∧
      invoke c B profile actual ctx before=invoke c B profile refOps ctx before := by
  have eq := laws.operations_eq actual refOps
  subst refOps
  have whole := whole_bytes c v A B requestedMax checked before observed storage labels profile actual ctx
  exact ⟨whole,whole,rfl⟩

#axiomsClean machine_facts
#axiomsClean whole_bytes
#axiomsClean whole_if_local
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Returned
