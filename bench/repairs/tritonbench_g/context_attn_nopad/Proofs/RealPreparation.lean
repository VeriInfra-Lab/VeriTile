import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideSource
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Wrapper
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.DirectOutput

/-! Independent Real preparation bridge. The ordered ordinary API establishes
coordinate bindings one role at a time, then the reused Real source proof
constructs the complete mathematical result. No floating graph or FP reader is
imported. Prefix errors retain their actual reached state and caller frame. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.RealPreparation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Logical HostPlan
noncomputable section

structure Context where
  kernel : BlockState
  caller : Nat → ℝ
  maximum : Nat
  maximumTensor : Nat
  itemValue : Nat
  resolvedStart : Nat → Int
  resolvedLength : Nat → Int
  outputObjectId : Nat
  outputAllocationId : Nat

def role (layout : Tensor3) (physical : Nat → ℝ) (logical : Nat → Nat → Nat → ℝ) : Prop :=
  ∀ row head dim, row < layout.rows → head < layout.heads → dim < layout.dims →
    physical (layout.offset row head dim) = logical row head dim

def binding (c : Call) (r : Regions) (A : FlatAlloc) (input : Logical.Inputs)
    (view : OutputView c) : Wrapper.Stage → Context → Prop
  | .resolveQ,s => role c.q (inputs s.kernel r).q input.query
  | .resolveK,s => role c.k (inputs s.kernel r).k input.key
  | .resolveV,s => role c.v (inputs s.kernel r).v input.value
  | .resolveStart,s => ∀ b, b < c.batches → s.resolvedStart b = (c.start b : Int)
  | .resolveLength,s => ∀ b, b < c.batches → s.resolvedLength b = (c.length b : Int)
  | .contiguousStart,s => ∀ b, b < c.batches → (inputs s.kernel r).start b = s.resolvedStart b
  | .contiguousLength,s => ∀ b, b < c.batches → (inputs s.kernel r).sequence b = s.resolvedLength b
  | .resolveOut,s => FlatInitialized c r A s.kernel view s.caller ∧
      (s.outputObjectId = view.objectId → view.negative = Bool.false ∧
        DirectOutput.Binding c r A view.address view.allocationId s.outputAllocationId
          (fun off => (A.flattenState s.kernel).readMem A.flat (A.addr r.out off)) s.caller)
  | .maximum,s => s.maximumTensor = metadataMaximum c r s.kernel
  | .item,s => s.itemValue = s.maximumTensor
  | .toPythonInt,s => s.maximum = s.itemValue
  | _,_ => True

structure API (c : Call) (r : Regions) (A : FlatAlloc) (input : Logical.Inputs) (view : OutputView c) where
  step : Wrapper.Stage → Context → Wrapper.StepResult Context
  establish : ∀ stage s t, step stage s = .ok t → binding c r A input view stage t
  preserve : ∀ stage s other, other ≠ stage → binding c r A input view other s →
    binding c r A input view other (step stage s).state
  frame : ∀ stage s addr, stage ≠ .deviceSetup → stage ≠ .copyback →
    (step stage s).state.caller addr = s.caller addr

def API.prefix {c : Call} {r : Regions} {A : FlatAlloc} {input : Logical.Inputs} {view : OutputView c}
    (api : API c r A input view) : Wrapper.API Context Nat ℝ where
  step := api.step
  observe := Context.caller
  caller := fun _ => True
  binds := binding c r A input view
  establish := api.establish
  preserve := api.preserve
  frame := fun stage s addr h1 h2 _ => api.frame stage s addr h1 h2

def Agrees (c : Call) (request : Wrapper.Request) : Prop :=
  request.maxInputLen = c.maxInputLen ∧ ∀ d, Wrapper.validate request = .ok d →
    d.width = c.q.dims ∧ d.batches = c.batches ∧ d.heads = c.q.heads

def synchronizedMaximum (d : Wrapper.Dimensions) (s : Context) : Nat :=
  if d.batches = 0 then 0 else s.maximum

/-- Constructed independent causal-attention success from the ordinary API's
successful prefix. The value returned by max/item equals the metadata value
used by the actual native stream. The original caller allocation is retained. -/
theorem ready_correct {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (input : Logical.Inputs) (view : OutputView c) (hinj : Function.Injective view.address)
    (api : API c r A input view) (request : Wrapper.Request)
    (agree : Agrees c request)
    (s t : Context) (d : Wrapper.Dimensions)
    (hp : Wrapper.runPrefix api.prefix request s = .ready d t) :
    t.caller = s.caller ∧
    synchronizedMaximum d t = metadataMaximum c r t.kernel ∧
    WideSource.Contract c r A t.kernel types hm.shape input view s.caller := by
  have hb := Wrapper.prefix_binds api.prefix request s t d hp
  have hs : ∀ b, b < c.batches → (inputs t.kernel r).start b = (c.start b : Int) :=
    fun b hb' => (hb.2 .contiguousStart (by simp [Wrapper.normalization]) b hb').trans
      (hb.2 .resolveStart (by simp [Wrapper.normalization]) b hb')
  have hl : ∀ b, b < c.batches → (inputs t.kernel r).sequence b = (c.length b : Int) :=
    fun b hb' => (hb.2 .contiguousLength (by simp [Wrapper.normalization]) b hb').trans
      (hb.2 .resolveLength (by simp [Wrapper.normalization]) b hb')
  have hn : Normalizes c (inputs t.kernel r) input :=
    ⟨hb.2 .resolveQ (by simp [Wrapper.normalization]),
     hb.2 .resolveK (by simp [Wrapper.normalization]),
     hb.2 .resolveV (by simp [Wrapper.normalization])⟩
  have hi : FlatInitialized c r A t.kernel view t.caller := (hb.2 .resolveOut (by simp [Wrapper.normalization])).1
  have hc : t.caller = s.caller := by
    funext addr
    have hf := Wrapper.prefix_frame api.prefix request s addr trivial
    change (Wrapper.runPrefix api.prefix request s).state.caller addr = s.caller addr at hf
    simpa only [hp,Wrapper.PrefixResult.state] using hf
  have memory : WideDomain.Memory c r A t.kernel :=
    ⟨hm.shape,hs,hl,hm.startExtent,hm.lengthExtent,hm.qBounded,hm.kBounded,hm.vBounded,
      hm.oBounded,hm.outQuery,hm.outKey,hm.outValue,hm.outStart,hm.outSequence⟩
  refine ⟨hc,?_,?_⟩
  · rw [WideDomain.metadataMaximum_eq memory]
    by_cases hz : d.batches = 0
    · have hzero : c.batches = 0 := (agree.2 d hb.1).2.1.symm.trans hz
      have hm0 : maximum c = 0 := by
        apply Nat.eq_zero_of_le_zero
        apply Finset.sup_le
        intro b _
        have := b.isLt
        omega
      simp [synchronizedMaximum,hz,hm0]
    · have hn := hb.2 .toPythonInt (by simp [Wrapper.synchronization,hz])
      have hi := hb.2 .item (by simp [Wrapper.synchronization,hz])
      have hm := hb.2 .maximum (by simp [Wrapper.synchronization,hz])
      have he : t.maximum = metadataMaximum c r t.kernel := hn.trans (hi.trans hm)
      rw [WideDomain.metadataMaximum_eq memory] at he
      exact (if_neg hz).trans he
  · rw [hc] at hi
    exact WideSource.whole_entry_correct memory ha types fits alloc hd hcov input hn view hinj s.caller hi

/-- The direct branch reads the final actual flat allocation. -/
def directFinish (c : Call) (r : Regions) (A : FlatAlloc) (t : Context) (raw : BlockState) : CallerResult c :=
  ⟨t.outputObjectId,t.outputAllocationId,
    DirectOutput.observe r A t.caller (fun off => raw.readMem A.flat (A.addr r.out off)),none⟩

theorem direct_finish_eq {c : Call} {r : Regions} {A : FlatAlloc} {t : Context}
    (hm : WideDomain.Memory c r A t.kernel) (view : OutputView c) (raw : BlockState)
    (same : t.outputObjectId = view.objectId) (negative : view.negative = Bool.false)
    (binding : DirectOutput.Binding c r A view.address view.allocationId t.outputAllocationId
      (fun off => (A.flattenState t.kernel).readMem A.flat (A.addr r.out off)) t.caller)
    (frame : BlockState.WriteWithin ((effective c r t.kernel).writes r A t.kernel)
      (A.flattenState t.kernel) raw) :
    directFinish c r A t raw = finish view t.caller (Values c r A raw) := by
  have memory := DirectOutput.observe_eq_writeback view.address
    (fun off => (A.flattenState t.kernel).readMem A.flat (A.addr r.out off)) t.caller
    (fun off => raw.readMem A.flat (A.addr r.out off)) binding.placement
    (fun i => hm.oBounded i.1.val i.2.1.val i.2.2.val i.1.isLt i.2.1.isLt i.2.2.isLt)
    binding.initialMemory (fun off hnot => by
      have hn : ¬ (effective c r t.kernel).writes r A t.kernel (A.flat,A.addr r.out off) := by
        intro hw
        exact hnot (DirectOutput.write_has_coordinate (WideNative.capMemory hm _) off hw)
      change raw.readMem A.flat (A.addr r.out off) = (A.flattenState t.kernel).readMem A.flat (A.addr r.out off)
      unfold BlockState.readMem
      rw [← frame A.flat (A.addr r.out off) hn])
  have he : DirectOutput.observe r A t.caller (fun off => raw.readMem A.flat (A.addr r.out off)) =
      view.writeback t.caller (Values c r A raw) := by
    unfold OutputView.writeback
    simpa only [Values,signed,negative,Bool.false_eq_true,↓reduceIte] using memory
  simp only [directFinish,finish,same,binding.allocation,he]

/-- Actual direct-allocation observation with the independent causal target. -/
def DirectContract (c : Call) (r : Regions) (A : FlatAlloc) (t : Context)
    (types : WideNative.MetadataTypes) (shape : WideDomain.Shape c)
    (input : Logical.Inputs) (view : OutputView c) (before : Nat → ℝ) : Prop :=
  ∃ raw, WideSource.Returns c r A t.kernel types raw ∧
    let result := directFinish c r A t raw.1
    result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧
    result.pythonReturn = none ∧
    (∀ b h q d hb hh ht hd, q < c.blocks*128 →
      view.read result.memory (WideLogical.outCoord shape b h q d hb hh ht hd) = expected c input b h q d) ∧
    (∀ addr, (¬ ∃ i, Written c i ∧ view.address i = addr) → result.memory addr = before addr)

theorem ready_direct {c : Call} {r : Regions} {A : FlatAlloc} {layoutState : BlockState}
    (hm : WideDomain.Memory c r A layoutState) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (input : Logical.Inputs) (view : OutputView c) (hinj : Function.Injective view.address)
    (api : API c r A input view) (request : Wrapper.Request) (agree : Agrees c request)
    (s t : Context) (d : Wrapper.Dimensions)
    (hp : Wrapper.runPrefix api.prefix request s = .ready d t)
    (same : t.outputObjectId = view.objectId) :
    DirectContract c r A t types hm.shape input view s.caller := by
  obtain ⟨caller,_,raw,returns,object,allocation,noneValue,output,frame,flat,inputs⟩ :=
    ready_correct hm ha types fits alloc hd hcov input view hinj api request agree s t d hp
  have hb := (Wrapper.prefix_binds api.prefix request s t d hp).2
  have hs : ∀ b, b < c.batches → (Program.inputs t.kernel r).start b = (c.start b : Int) :=
    fun b hb' => (hb .contiguousStart (by simp [Wrapper.normalization]) b hb').trans
      (hb .resolveStart (by simp [Wrapper.normalization]) b hb')
  have hl : ∀ b, b < c.batches → (Program.inputs t.kernel r).sequence b = (c.length b : Int) :=
    fun b hb' => (hb .contiguousLength (by simp [Wrapper.normalization]) b hb').trans
      (hb .resolveLength (by simp [Wrapper.normalization]) b hb')
  have memory : WideDomain.Memory c r A t.kernel :=
    ⟨hm.shape,hs,hl,hm.startExtent,hm.lengthExtent,hm.qBounded,hm.kBounded,hm.vBounded,
      hm.oBounded,hm.outQuery,hm.outKey,hm.outValue,hm.outStart,hm.outSequence⟩
  obtain ⟨negative,binding⟩ := (hb .resolveOut (by simp [Wrapper.normalization])).2 same
  have eq := direct_finish_eq memory view raw.1 same negative binding flat
  refine ⟨raw,returns,?_⟩
  dsimp only
  rw [eq,caller]
  exact ⟨object,allocation,noneValue,output,frame⟩

/-- Prefix failure states obey the same caller frame as successful preparation;
normalization errors precede assertions and synchronization errors follow them. -/
theorem prefix_frame {c : Call} {r : Regions} {A : FlatAlloc}
    {input : Logical.Inputs} {view : OutputView c} (api : API c r A input view)
    (request : Wrapper.Request) (s : Context) :
    (Wrapper.runPrefix api.prefix request s).state.caller = s.caller := by
  funext addr
  exact Wrapper.prefix_frame api.prefix request s addr trivial

#print axioms direct_finish_eq
#print axioms ready_direct
#print axioms ready_correct
#print axioms prefix_frame
end
end VeriTile.Bench.Repairs.ContextAttnNopad.RealPreparation
