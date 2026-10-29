import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.HostNative
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ReturnCopies

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open ByteMemory HostStorage RawStorage SourceAddress
noncomputable section
set_option maxHeartbeats 8000000

structure InnerBinding (r : Admission.Inner) (before : Memory) (p : InnerPrepared r before)
    (scratch : State NativeGraph.Expr) : Prop where
  key : ∀ i,i<keyCapacity r → ((HostData.innerInput r before scratch).mem "k" i).read .real=
    .leaf (read p.memory (p.key.byteBase+4*i))
  value : ∀ i,i<valueCapacity r → ((HostData.innerInput r before scratch).mem "v" i).read .real=
    .leaf (read p.memory (p.value.byteBase+4*i))
  gate : ∀ i,i<gateCapacity r → ((HostData.innerInput r before scratch).mem "g" i).read .real=
    .leaf (read p.memory (p.gate.byteBase+4*i))
  seed : r.seed.isSome=Bool.true → ∀ i,i<finalCapacity r →
    ((HostData.innerInput r before scratch).mem "h0" i).read .fp32=.leaf (read p.memory (p.seed.byteBase+4*i))
  pointers : ∀ (i : InnerCoord r.config) site,enabled r.config site →
    (innerWord r.config i site).toInt=(innerOffset r.config i site : Int) ∧
    (innerWord r.config i site*Address.word 4).toInt=(4*innerOffset r.config i site : Nat) ∧
    (BitVec.ofNat 64 (HostNative.siteTensor p site).byteBase+innerWord r.config i site*Address.word 4).toNat=
      (HostNative.siteTensor p site).byteBase+4*innerOffset r.config i site ∧
    ((HostNative.siteTensor p site).byteBase+4*innerOffset r.config i site)%4=0 ∧
    (HostNative.siteTensor p site).byteBase+4*innerOffset r.config i site+4<2^64
  linear : ∀ job∈r.config.plans,∀ lane : Fin job.size,
    (Address.word lane.val+Address.word job.base).toInt=(job.base+lane.val : Nat)

theorem inner_binding (r : Admission.Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (before : Memory) (p : InnerPrepared r before) (pointers : p.check=Bool.true) (scratch : State NativeGraph.Expr) :
    InnerBinding r before p scratch :=
  ⟨key_load r before p scratch,value_load r before p scratch,gate_load r before p scratch,
    seed_load r before p scratch,HostNative.inner r checked nonempty before p pointers,
    CheckedNative.linear r checked nonempty⟩

structure PrefixBinding (r : Admission.Prefix) (before : Memory) (p : PrefixPrepared r before)
    (BS : Nat) (scratch : State NativeGraph.Expr) : Prop where
  input : ∀ i,i<prefixCapacity r → ((HostData.prefixInput r before scratch).mem "s" i).read .fp32=
    .leaf (read p.memory (p.input.byteBase+4*i))
  inputPointer : ∀ i : SourceCumulative.Coord (r.config BS),
    (BitVec.ofNat 64 p.input.byteBase+SourceCumulative.nativeWord (r.config BS) i*Address.word 4).toNat=
      p.input.byteBase+4*SourceCumulative.offset (r.config BS) i ∧
    (p.input.byteBase+4*SourceCumulative.offset (r.config BS) i)%4=0 ∧
    p.input.byteBase+4*SourceCumulative.offset (r.config BS) i+4<2^64
  outputPointer : ∀ i : SourceCumulative.Coord (r.config BS),
    (BitVec.ofNat 64 p.output.byteBase+SourceCumulative.nativeWord (r.config BS) i*Address.word 4).toNat=
      p.output.byteBase+4*SourceCumulative.offset (r.config BS) i ∧
    (p.output.byteBase+4*SourceCumulative.offset (r.config BS) i)%4=0 ∧
    p.output.byteBase+4*SourceCumulative.offset (r.config BS) i+4<2^64
  guard : ∀ job∈(r.config BS).plans,∀ lane : Fin job.size,
    ((Address.word lane.val+Address.word job.base).toInt<(Address.word (r.config BS).total).toInt ↔
      job.base+lane.val<(r.config BS).total)

theorem prefix_binding (r : Admission.Prefix) (checked : r.check=Bool.true) (nonempty : r.nonempty)
    (BS : Nat) (selected : CheckedPrefixNative.Selected BS) (before : Memory) (p : PrefixPrepared r before)
    (pointers : p.check=Bool.true) (scratch : State NativeGraph.Expr) : PrefixBinding r before p BS scratch :=
  ⟨prefix_load r before p scratch,
    fun i=>(HostNative.prefix_input r checked nonempty BS selected before p pointers i).2.2,
    fun i=>(HostNative.prefix_output r checked nonempty BS selected before p pointers i).2.2,
    CheckedPrefixNative.guard r checked nonempty BS selected⟩

def finalValue (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (scratch : State NativeGraph.Expr) (i : Nat) : Word :=
  if r.nonempty then HostData.finalWord I r before scratch i else HostData.emptyFinal I r before i

/-- One successful checked host, its complete derived source grid, original
byte-bound input loads, native pointer guards, returned bytes and caller frame. -/
structure InnerResult (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (rt : CheckedHost.InnerRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat)
    (result : CheckedHost.Returned) : Prop where
  host : CheckedHost.InnerSuccess I r before rt scratch caller result
  checked : r.check=Bool.true
  metadata : result.tensor.Valid r.query.device [r.B,r.H,r.config.NT*r.K,r.V]
  restored : result.currentDevice=caller
  owner : result.launchDevice=(if r.nonempty then some r.query.device else none)
  history : ∀ i,i < innerCount r → read result.memory (result.tensor.byteBase+4*i)=HostData.historyWord I r before scratch i
  final : ∀ t,r.final=some t → ∀ i,i<finalCount r → read result.memory (t.byteBase+4*i)=finalValue I r before scratch i
  callerFrame : ∀ t∈r.tensors,∀ j,j<4*t.shape.prod →
    (∀ output,r.final=some output → Outside output.byteBase (finalCount r) (t.byteBase+j)) →
    result.memory (t.byteBase+j)=before (t.byteBase+j)
  source : HostFP.batches NativeValue.algebra (NativeGrid.program (NativeGrid.innerSource r.config))
    (NativeGrid.innerOwns r.config) r.config.plans (HostData.innerInput r before scratch)=some (HostData.innerResult r before scratch)
  bindings : r.nonempty → ∃ p,rt.prepare=.ok p ∧ p.check=Bool.true ∧ InnerBinding r before p scratch

theorem inner_returned (I : ScalarNative.Primitives) (r : Admission.Inner) (before : Memory)
    (rt : CheckedHost.InnerRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat)
    (result : CheckedHost.Returned) (success : CheckedHost.runInner I r before rt scratch caller=.ok result) :
    InnerResult I r before rt scratch caller result := by
  have post:=CheckedHost.inner_complete I r before rt scratch caller
  rw [success] at post
  cases post with
  | empty checked empty place allocation copy=>
    refine ⟨.empty checked empty place allocation copy,checked,place.valid,rfl,by simp [empty],?_,?_,?_,
      HostData.inner_source r before scratch,?_⟩
    · intro i hi
      have count:=HostCoordinates.empty_count r checked empty
      omega
    · intro t ht i hi
      simp only [finalValue,if_neg empty]
      exact ReturnedMemory.empty_read I r before t ht i hi
    · intro t ht j hj outside
      exact ReturnedMemory.copy_frame r before (HostData.emptyFinal I r before) _ outside
    · intro nonempty
      exact False.elim (empty nonempty)
  | active checked nonempty p allocation pointers setup finish returned=>
    refine ⟨.active checked nonempty p allocation pointers setup finish returned,checked,finish.place.valid,rfl,
      by simp [nonempty],ReturnedMemory.inner_read I r before p finish scratch,?_,
      ReturnedMemory.inner_caller_frame I r before p finish scratch,HostData.inner_source r before scratch,
      fun _=>⟨p,allocation,pointers,inner_binding r checked nonempty before p pointers scratch⟩⟩
    intro t ht i hi
    simp only [finalValue,if_pos nonempty]
    exact ReturnedMemory.final_read I r before p finish scratch t ht i hi (r.final_valid checked t ht).2.2.2.2.2.2.2.2

structure PrefixResult (I : ScalarNative.Primitives) (r : Admission.Prefix) (before : Memory)
    (rt : CheckedHost.PrefixRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat)
    (result : CheckedHost.Returned) : Prop where
  host : CheckedHost.PrefixSuccess I r before rt scratch caller result
  checked : r.check=Bool.true
  metadata : result.tensor.Valid r.input.device [r.B,r.H,r.T,r.S]
  restored : result.currentDevice=caller
  owner : result.launchDevice=(if r.nonempty then some r.input.device else none)
  values : ∀ i,i<prefixCount r → read result.memory (result.tensor.byteBase+4*i)=HostData.prefixWord I r rt.BS before scratch i
  callerFrame : ∀ j,j<4*r.input.shape.prod → result.memory (r.input.byteBase+j)=before (r.input.byteBase+j)
  source : HostFP.batches NativeValue.algebra (NativeGrid.program (NativeGrid.prefixSource (r.config rt.BS)))
    (NativeGrid.prefixOwns (r.config rt.BS)) (r.config rt.BS).plans (HostData.prefixInput r before scratch)=
      some (HostData.prefixResult r rt.BS before scratch)
  bindings : r.nonempty → ∃ p,rt.prepare=.ok p ∧ p.check=Bool.true ∧ PrefixBinding r before p rt.BS scratch

theorem prefix_returned (I : ScalarNative.Primitives) (r : Admission.Prefix) (before : Memory)
    (rt : CheckedHost.PrefixRuntime r before) (scratch : State NativeGraph.Expr) (caller : Nat)
    (result : CheckedHost.Returned) (success : CheckedHost.runPrefix I r before rt scratch caller=.ok result) :
    PrefixResult I r before rt scratch caller result := by
  have post:=CheckedHost.prefix_complete I r before rt scratch caller
  rw [success] at post
  cases post with
  | empty checked empty place allocation=>
    refine ⟨.empty checked empty place allocation,checked,place.valid,rfl,by simp [empty],?_,
      fun _ _=>rfl,HostData.prefix_source r rt.BS before scratch,?_⟩
    · intro i hi
      have count : prefixCount r=0:=not_not.mp empty
      omega
    · intro nonempty
      exact False.elim (empty nonempty)
  | active checked nonempty p allocation pointers setup finish returned=>
    exact ⟨.active checked nonempty p allocation pointers setup finish returned,checked,finish.place.valid,rfl,
      by simp [nonempty],ReturnedMemory.prefix_read I r rt.BS before p finish scratch,
      ReturnedMemory.prefix_caller_frame I r rt.BS before p finish scratch,HostData.prefix_source r rt.BS before scratch,
      fun _=>⟨p,allocation,pointers,prefix_binding r checked nonempty rt.BS rt.selected before p pointers scratch⟩⟩

abbrev checked_inner := @CheckedHost.inner_complete
abbrev checked_prefix := @CheckedHost.prefix_complete
abbrev native_range := @CheckedNative.range
abbrev exterior_inner := @ReturnedMemory.inner_frame
abbrev exterior_prefix := @ReturnedMemory.prefix_frame
abbrev local_numerical_transport := @ScalarNative.transport
abbrev shared_inner_return := @ReturnCopies.inner_shared
abbrev shared_prefix_return := @ReturnCopies.prefix_shared
#axiomsClean inner_returned
#axiomsClean prefix_returned
#axiomsClean native_range
#axiomsClean exterior_inner
#axiomsClean exterior_prefix
#axiomsClean local_numerical_transport
#axiomsClean shared_inner_return
#axiomsClean shared_prefix_return
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.FPEquiv
