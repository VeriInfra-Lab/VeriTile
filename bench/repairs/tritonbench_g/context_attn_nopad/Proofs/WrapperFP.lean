import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Launch
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Copyback
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.DirectOutput

/-! Actual ordered wrapper composition. Ordinary API steps establish only
coordinate bindings, exact metadata/max values and tensor identity. Successful
kernel execution and full frames are derived from the source proof. Every
failure retains the reached prefix and state; copyback runs only after launch.
The actual returned max/item value selects the effective call. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.WrapperFP
open VeriTile.Triton FP.Structural
open FPHostSource
noncomputable section
set_option maxHeartbeats 3000000

structure LogicalInputs (α : Type) where
  query : Nat → Nat → Nat → α
  key : Nat → Nat → Nat → α
  value : Nat → Nat → Nat → α

structure Context (α : Type) where
  kernel : State α
  caller : Nat → α
  maximum : Nat
  maximumTensor : Nat
  itemValue : Nat
  resolvedStart : Nat → Int
  resolvedLength : Nat → Int
  /-- Identity of the tensor actually returned by output resolve_neg. -/
  outputObjectId : Nat
  /-- Allocation owning the actual kernel r.out pointer after resolve_neg. -/
  outputAllocationId : Nat

def Context.copied {α : Type} {c : Call} (s : Context α) (view : OutputView α c) : Bool :=
  decide (s.outputObjectId ≠ view.objectId)

theorem copy_condition {α : Type} {c : Call} (s : Context α) (view : OutputView α c) :
    s.copied view = Bool.true ↔ s.outputObjectId ≠ view.objectId := by simp [Context.copied]

def role {α : Type} [Inhabited α] (layout : VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Tensor3) (region : RegionName)
    (values : Nat → Nat → Nat → α) (s : State α) : Prop :=
  ∀ row head dim, row < layout.rows → head < layout.heads → dim < layout.dims →
    (s.mem region (layout.offset row head dim)).read .real = values row head dim

def binding {α : Type} [Inhabited α] (c : Call) (r : Regions) (A : FlatAlloc)
    (input : LogicalInputs α) (view : OutputView α c) : Wrapper.Stage → Context α → Prop
  | .resolveQ,s => role c.q r.query input.query s.kernel
  | .resolveK,s => role c.k r.key input.key s.kernel
  | .resolveV,s => role c.v r.value input.value s.kernel
  | .resolveStart,s => ∀ b, b < c.batches → s.resolvedStart b = (c.start b : Int)
  | .resolveLength,s => ∀ b, b < c.batches → s.resolvedLength b = (c.length b : Int)
  | .contiguousStart,s => ∀ b, b < c.batches →
      (FPSymbolic.integerInputs r s.kernel).start b = s.resolvedStart b
  | .contiguousLength,s => ∀ b, b < c.batches →
      (FPSymbolic.integerInputs r s.kernel).sequence b = s.resolvedLength b
  | .resolveOut,s => FPHostPrecise.Initialized c r s.kernel view s.caller ∧
      (s.outputObjectId = view.objectId → view.action = id ∧
        DirectOutput.Binding c r A view.address view.allocationId s.outputAllocationId
          (fun off => (s.kernel.mem r.out off).read .real) s.caller)
  | .maximum,s => s.maximumTensor = metadataMaximum c r s.kernel
  | .item,s => s.itemValue = s.maximumTensor
  | .toPythonInt,s => s.maximum = s.itemValue
  | _,_ => True

/-- Coordinate-preserving resolve_neg/contiguous and exact max/item contracts.
No execution, output formula or successful kernel result is an API field. -/
structure Preparation (α : Type) [Inhabited α] (c : Call) (r : Regions) (A : FlatAlloc)
    (input : LogicalInputs α) (view : OutputView α c) where
  step : Wrapper.Stage → Context α → Wrapper.StepResult (Context α)
  establish : ∀ stage s t, step stage s = .ok t → binding c r A input view stage t
  preserve : ∀ stage s other, other ≠ stage → binding c r A input view other s →
    binding c r A input view other (step stage s).state
  frame : ∀ stage s addr, stage ≠ .deviceSetup → stage ≠ .copyback →
    (step stage s).state.caller addr = s.caller addr

def Preparation.api {α : Type} [Inhabited α] {c : Call} {r : Regions}
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view) :
    Wrapper.API (Context α) Nat α where
  step := p.step
  observe := Context.caller
  caller := fun _ => True
  binds := binding c r A input view
  establish := p.establish
  preserve := p.preserve
  frame := fun stage s addr h1 h2 _ => p.frame stage s addr h1 h2

structure Prepared {α : Type} [Inhabited α] (c : Call) (r : Regions)
    (input : LogicalInputs α) (view : OutputView α c) (s : Context α) : Prop where
  metadata : Metadata c r s.kernel
  query : role c.q r.query input.query s.kernel
  key : role c.k r.key input.key s.kernel
  value : role c.v r.value input.value s.kernel
  initialized : FPHostPrecise.Initialized c r s.kernel view s.caller

def Agrees (c : Call) (request : Wrapper.Request) : Prop :=
  request.maxInputLen = c.maxInputLen ∧ ∀ d, Wrapper.validate request = .ok d →
    d.width = c.q.dims ∧ d.batches = c.batches ∧ d.heads = c.q.heads

def synchronizedMaximum {α : Type} (d : Wrapper.Dimensions) (s : Context α) : Nat :=
  if d.batches = 0 then 0 else s.maximum

theorem prefix_prepared {α : Type} [Inhabited α] {c : Call} {r : Regions}
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (request : Wrapper.Request) (s t : Context α) (d : Wrapper.Dimensions)
    (h : Wrapper.runPrefix p.api request s = .ready d t) : Prepared c r input view t := by
  have hb := (Wrapper.prefix_binds p.api request s t d h).2
  refine ⟨⟨?_,?_⟩,?_,?_,?_,?_⟩
  · intro b hb'
    exact (hb .contiguousStart (by simp [Wrapper.normalization]) b hb').trans
      (hb .resolveStart (by simp [Wrapper.normalization]) b hb')
  · intro b hb'
    exact (hb .contiguousLength (by simp [Wrapper.normalization]) b hb').trans
      (hb .resolveLength (by simp [Wrapper.normalization]) b hb')
  · exact hb .resolveQ (by simp [Wrapper.normalization])
  · exact hb .resolveK (by simp [Wrapper.normalization])
  · exact hb .resolveV (by simp [Wrapper.normalization])
  · exact (hb .resolveOut (by simp [Wrapper.normalization])).1

theorem prefix_caller {α : Type} [Inhabited α] {c : Call} {r : Regions}
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (request : Wrapper.Request) (s t : Context α) (d : Wrapper.Dimensions)
    (h : Wrapper.runPrefix p.api request s = .ready d t) : t.caller = s.caller := by
  funext addr
  have hf := Wrapper.prefix_frame p.api request s addr trivial
  change (Wrapper.runPrefix p.api request s).state.caller addr = s.caller addr at hf
  simpa only [h,Wrapper.PrefixResult.state] using hf

/-- Every prefix outcome, including failure, preserves the caller allocation. -/
theorem prefix_all_frame {α : Type} [Inhabited α] {c : Call} {r : Regions}
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (request : Wrapper.Request) (s : Context α) :
    (Wrapper.runPrefix p.api request s).state.caller = s.caller := by
  funext addr
  exact Wrapper.prefix_frame p.api request s addr trivial

theorem actual_maximum {α : Type} [Inhabited α] {c : Call} {r : Regions}
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (request : Wrapper.Request) (agree : Agrees c request) (s t : Context α) (d : Wrapper.Dimensions)
    (h : Wrapper.runPrefix p.api request s = .ready d t) :
    synchronizedMaximum d t = metadataMaximum c r t.kernel := by
  have hb := Wrapper.prefix_binds p.api request s t d h
  have hd := (agree.2 d hb.1).2.1
  rw [metadataMaximum_eq (prefix_prepared p request s t d h).metadata]
  by_cases hz : d.batches = 0
  · have hc : c.batches = 0 := hd.symm.trans hz
    have hm : HostPlan.maximum c = 0 := by
      apply Nat.eq_zero_of_le_zero
      apply Finset.sup_le
      intro b _
      have := b.isLt
      omega
    simp [synchronizedMaximum,hz,hm]
  · have hp := hb.2 .toPythonInt (by simp [Wrapper.synchronization,hz])
    have hi := hb.2 .item (by simp [Wrapper.synchronization,hz])
    have hm := hb.2 .maximum (by simp [Wrapper.synchronization,hz])
    have he : t.maximum = metadataMaximum c r t.kernel := hp.trans (hi.trans hm)
    rw [metadataMaximum_eq (prefix_prepared p request s t d h).metadata] at he
    exact (if_neg hz).trans he

inductive Result (α : Type) where
  | invalid (error : Wrapper.ValidationError) (state : Context α)
  | prefixFailure (stage : Wrapper.Stage) (error : String) (state : Context α)
  | launchFailure (dimensions : Wrapper.Dimensions) (prepared : Context α)
      (chunk : Launch.Chunk) (error : String) (raw : State α)
  | copyFailure (dimensions : Wrapper.Dimensions) (prepared : Context α)
      (raw : State α) (error : String) (memory : Nat → α)
  | ok (dimensions : Wrapper.Dimensions) (prepared : Context α) (raw : State α) (result : FPHostSource.Result α)
  | executionFailure

def directFinish {α : Type} [Inhabited α] (r : Regions) (A : FlatAlloc)
    (t : Context α) (raw : State α) : FPHostSource.Result α :=
  ⟨t.outputObjectId,t.outputAllocationId,
    DirectOutput.observe r A t.caller (fun off => (raw.mem r.out off).read .real),none⟩

def run {α : Type} [Inhabited α] (o : FPBackend.Ops α) {c : Call} {r : Regions}
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (launch : Launch.API α) (copy : Copyback.API α c view)
    (request : Wrapper.Request) (s : Context α) : Result α :=
  match Wrapper.runPrefix p.api request s with
  | .invalid e t => .invalid e t
  | .failed stage e t => .prefixFailure stage e t
  | .ready d t =>
    let ec := NativeIndex.capCall c (synchronizedMaximum d t)
    match Launch.run o launch ec r (HostPlan.jobs ec) none t.kernel with
    | .executionFailure => .executionFailure
    | .failed base e raw => .launchFailure d t base e raw
    | .ok raw =>
      if t.copied view then
        match copy.copy (fun i => (raw.mem r.out (Logical.coordOffset c i)).read .real) t.caller with
        | .failed e mem => .copyFailure d t raw e mem
        | .ok mem => .ok d t raw ⟨view.objectId,view.allocationId,mem,none⟩
      else .ok d t raw (directFinish r A t raw)

/-- The successful raw stream and its complete caller/flat frame. This is
proved from the typed source; it is never an ordinary API hypothesis. -/
def RawSuccess {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (c : Call) (r : Regions) (A : FlatAlloc) (layoutState : BlockState)
    (view : OutputView α c) (t : Context α) (raw : State α) : Prop :=
  runHost o.algebra c r t.kernel view t.caller = some (finish r view t.caller raw) ∧
  reference o.algebra c r t.kernel view t.caller = some (finish r view t.caller raw) ∧
  (∀ i, view.action ((finish r view t.caller raw).memory (view.address i)) =
    (raw.mem r.out (Logical.coordOffset c i)).read .real) ∧
  (∀ addr, (¬ ∃ i, Logical.Written c i ∧ view.address i = addr) →
    (finish r view t.caller raw).memory addr = t.caller addr) ∧
  (∀ rg off, ¬ (effective c r t.kernel).writes r A layoutState (A.flat,A.addr rg off) →
    raw.mem rg off = t.kernel.mem rg off)

theorem raw_success {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    {types : WideNative.MetadataTypes} (fits : WideNative.MetadataFits c types)
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (launch : Launch.API α) (request : Wrapper.Request) (agree : Agrees c request)
    (s t : Context α) (d : Wrapper.Dimensions) (raw : State α)
    (hp : Wrapper.runPrefix p.api request s = .ready d t)
    (hl : Launch.run o launch (NativeIndex.capCall c (synchronizedMaximum d t)) r
      (HostPlan.jobs (NativeIndex.capCall c (synchronizedMaximum d t))) none t.kernel = .ok raw) :
    RawSuccess o c r A layoutState view t raw := by
  have prep := prefix_prepared p request s t d hp
  have he : NativeIndex.capCall c (synchronizedMaximum d t) = effective c r t.kernel := by
    rw [actual_maximum p request agree s t d hp]
    rfl
  have hs := Launch.successful_stream o launch _ r _ none t.kernel raw hl
  rw [he] at hs
  obtain ⟨raw',result,hr,hh,href,hfinish,_,_,_,hread,hframe,hflat⟩ :=
    WideFP.whole_host_precise o hm ha fits prep.metadata view t.caller prep.initialized
  have eq : raw' = raw := Option.some.inj (hr.symm.trans hs)
  subst raw'
  subst result
  exact ⟨hh,href,hread,hframe,hflat⟩

/-- The no-copy branch observes the same physical allocation. Its equality
with the mathematical writeback is derived from the actual source frame. -/
theorem direct_finish_eq {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : WideDomain.Memory c r A layoutState)
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (request : Wrapper.Request) (s t : Context α) (d : Wrapper.Dimensions) (raw : State α)
    (hp : Wrapper.runPrefix p.api request s = .ready d t)
    (same : t.copied view = Bool.false)
    (hr : RawSuccess o c r A layoutState view t raw) :
    directFinish r A t raw = finish r view t.caller raw := by
  have hid : t.outputObjectId = view.objectId := by
    simpa [Context.copied] using same
  have hb := (Wrapper.prefix_binds p.api request s t d hp).2 .resolveOut
    (by simp [Wrapper.normalization])
  obtain ⟨action,placement⟩ := hb.2 hid
  have he := DirectOutput.observe_eq_writeback view.address
    (fun off => (t.kernel.mem r.out off).read .real) t.caller
    (fun off => (raw.mem r.out off).read .real) placement.placement
    (fun i => hm.oBounded i.1.val i.2.1.val i.2.2.val i.1.isLt i.2.1.isLt i.2.2.isLt)
    placement.initialMemory (fun off hnot => by
      have hn : ¬ (effective c r t.kernel).writes r A layoutState (A.flat,A.addr r.out off) := by
        intro hw
        exact hnot (DirectOutput.write_has_coordinate (WideNative.capMemory hm _) off hw)
      exact congrArg (fun value => value.read .real) (hr.2.2.2.2 r.out off hn))
  have memory : DirectOutput.observe r A t.caller (fun off => (raw.mem r.out off).read .real) =
      FPHostSource.writeback r view t.caller raw := by
    unfold FPHostSource.writeback
    simpa only [action,id_eq] using he
  simp only [directFinish,finish,hid,placement.allocation,memory]

/-- The result classification includes exact successful prefixes for errors,
constructed complete source execution for success, and a partial-copy frame.
Caller identities and Python None are retained by the actual returned object. -/
def Contract {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {input : LogicalInputs α} {view : OutputView α c}
    (p : Preparation α c r A input view) (launch : Launch.API α) (copy : Copyback.API α c view)
    (layoutState : BlockState) (request : Wrapper.Request) (s : Context α) : Prop :=
  match run o p launch copy request s with
  | .invalid e t => Wrapper.steps p.api Wrapper.normalization s = .ok t ∧ Wrapper.validate request = .error e
  | .prefixFailure stage e t => Wrapper.FailedAt p.api Wrapper.normalization s stage e t ∨
      ∃ u d, Wrapper.steps p.api Wrapper.normalization s = .ok u ∧ Wrapper.validate request = .ok d ∧
        Wrapper.FailedAt p.api (Wrapper.synchronization d.batches) u stage e t
  | .launchFailure d t base e raw =>
      Wrapper.runPrefix p.api request s = .ready d t ∧ t.caller = s.caller ∧
      Launch.FailedAt o launch (NativeIndex.capCall c (synchronizedMaximum d t)) r
        (HostPlan.jobs (NativeIndex.capCall c (synchronizedMaximum d t))) none t.kernel base e raw
  | .copyFailure d t raw e mem =>
      Wrapper.runPrefix p.api request s = .ready d t ∧ t.caller = s.caller ∧ t.copied view = Bool.true ∧
      RawSuccess o c r A layoutState view t raw ∧
      copy.copy (fun i => (raw.mem r.out (Logical.coordOffset c i)).read .real) t.caller = .failed e mem ∧
      ∀ addr, (¬ ∃ i, Logical.Written c i ∧ view.address i = addr) → mem addr = s.caller addr
  | .ok d t raw result =>
      Wrapper.runPrefix p.api request s = .ready d t ∧ t.caller = s.caller ∧
      RawSuccess o c r A layoutState view t raw ∧ result = finish r view s.caller raw ∧
      result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧ result.pythonReturn = none
  | .executionFailure => False

theorem whole_outcomes {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    {types : WideNative.MetadataTypes} (fits : WideNative.MetadataFits c types)
    {input : LogicalInputs α} {view : OutputView α c} (p : Preparation α c r A input view)
    (launch : Launch.API α) (copy : Copyback.API α c view)
    (request : Wrapper.Request) (agree : Agrees c request) (s : Context α) :
    Contract o p launch copy layoutState request s := by
  have protocol := Wrapper.prefix_protocol p.api request s
  cases hp : Wrapper.runPrefix p.api request s with
  | invalid e t => simpa only [Contract,run,hp] using (by simpa [hp] using protocol)
  | failed stage e t => simpa only [Contract,run,hp] using (by simpa [hp] using protocol)
  | ready d t =>
    have hc := prefix_caller p request s t d hp
    let ec := NativeIndex.capCall c (synchronizedMaximum d t)
    have lc := Launch.run_correct o launch ec r (HostPlan.jobs ec) none t.kernel
    cases hl : Launch.run o launch ec r (HostPlan.jobs ec) none t.kernel with
    | executionFailure => simp only [Launch.Contract,hl] at lc
    | failed base e raw =>
      dsimp only [ec] at hl
      simp only [Contract,run,hp,hl]
      exact ⟨(by trivial),hc,by simpa only [Launch.Contract,ec,hl] using lc⟩
    | ok raw =>
      dsimp only [ec] at hl
      have hr := raw_success o hm ha fits p launch request agree s t d raw hp hl
      cases copied : t.copied view with
      | «false» =>
        simp only [Contract,run,hp,hl,copied,Bool.false_eq_true,↓reduceIte]
        have hdirect := direct_finish_eq o hm p request s t d raw hp copied hr
        rw [hdirect]
        exact ⟨(by trivial),hc,hr,by rw [hc],(by trivial),(by trivial),(by trivial)⟩
      | «true» =>
        cases hcopy : copy.copy (fun i => (raw.mem r.out (Logical.coordOffset c i)).read .real) t.caller with
        | ok mem =>
          have hm' := Copyback.success_memory r view copy raw t.caller mem hcopy
          simp only [Contract,run,hp,hl,copied,↓reduceIte,hcopy]
          refine ⟨(by trivial),hc,hr,?_,(by trivial),(by trivial),(by trivial)⟩
          rw [hm',hc]
          rfl
        | failed e mem =>
          simp only [Contract,run,hp,hl,copied,↓reduceIte,hcopy]
          refine ⟨(by trivial),hc,(by trivial),hr,(by trivial),?_⟩
          intro addr hnot
          have stable : ∀ i, ¬ Logical.Written c i →
              (raw.mem r.out (Logical.coordOffset c i)).read .real = view.action (t.caller (view.address i)) := by
            intro i hi
            rw [← hr.2.2.1 i,hr.2.2.2.1 (view.address i) (by
              rintro ⟨j,hj,he⟩
              exact hi ((view.injective he) ▸ hj))]
          have hf := Copyback.outcome_frame view copy _ t.caller stable addr hnot
          rw [hcopy] at hf
          exact hf.trans (congrFun hc addr)

#print axioms prefix_prepared
#print axioms actual_maximum
#print axioms direct_finish_eq
#print axioms whole_outcomes
end
end VeriTile.Bench.Repairs.ContextAttnNopad.WrapperFP
