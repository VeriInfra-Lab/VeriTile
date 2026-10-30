import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RealPhysical

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Correct
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open HostStorage RawStorage
noncomputable section
set_option maxHeartbeats 8000000

def InnerSpec (r : Admission.Inner) (checked : r.check=Bool.true) (before : RealSource.Memory)
    (scratch : BlockState) (result : CheckedHost.Returned) : Prop :=
  (∃ place : InnerPlace r,result.objectId=place.objectId ∧ result.tensor=place.tensor) ∧
  (r.nonempty → DirectRealSequence.Runs (RealSource.layout r) RealSource.regions .fp32 .fp32 .fp32 .fp32 .fp32
    r.gateK r.seed.isSome r.final.isSome Bool.false r.config.plans
    {RealSource.inputState r before scratch with undef:=fun _ _=>0} (RealSource.innerResult r checked before scratch)) ∧
  (∀ i,i < innerCount r → RealPhysical.innerResult r checked before scratch result.tensor (result.tensor.byteBase+4*i)=
    RealSource.History r before i) ∧
  (∀ tensor,r.final=some tensor → ∀ i,i<finalCount r →
    RealPhysical.innerResult r checked before scratch result.tensor (tensor.byteBase+4*i)=RealSource.Final r before i) ∧
  (∀ address,Outside result.tensor.byteBase (innerCount r) address →
    (∀ tensor,r.final=some tensor → Outside tensor.byteBase (finalCount r) address) →
    RealPhysical.innerResult r checked before scratch result.tensor address=before address)

theorem inner_spec (r : Admission.Inner) (checked : r.check=Bool.true) (before : RealSource.Memory)
    (scratch : BlockState) (result : CheckedHost.Returned) (place : InnerPlace r)
    (identity : result.objectId=place.objectId) (tensor : result.tensor=place.tensor) :
    InnerSpec r checked before scratch result := by
  refine ⟨⟨place,identity,tensor⟩,?_,RealPhysical.history r checked before scratch result.tensor,?_,
    RealPhysical.inner_frame r checked before scratch result.tensor⟩
  · intro nonempty
    have source:=(RealSource.inner_exists r checked before scratch).choose_spec.1
    have he : r.config.empty=Bool.false := by simpa [HostFP.Inner.empty,Admission.Inner.config,Admission.Inner.nonempty] using nonempty
    simpa only [HostFP.Inner.plans,he,Bool.false_eq_true,if_false,HostFP.Inner.total,RealSource.innerResult] using source
  · rw [tensor]
    exact RealPhysical.final r checked before scratch place

/-- Real input interpretation is independent of FP32 bit payloads. The same
checked host identity receives the complete source's sliced history and final
state, expressed by independent valid-time contractions and gate products. -/
theorem inner_returned (I : ScalarNative.Primitives) (r : Admission.Inner) (checked : r.check=Bool.true)
    (bytes : ByteMemory.Memory) (rt : CheckedHost.InnerRuntime r bytes) (fpScratch : State NativeGraph.Expr)
    (caller : Nat) (result : CheckedHost.Returned)
    (success : CheckedHost.runInner I r bytes rt fpScratch caller=.ok result)
    (before : RealSource.Memory) (scratch : BlockState) : InnerSpec r checked before scratch result := by
  have post:=CheckedHost.inner_complete I r bytes rt fpScratch caller
  rw [success] at post
  cases post with
  | empty _ _ place _ _=>exact inner_spec r checked before scratch _ place rfl rfl
  | active _ _ _ _ _ _ finish _=>exact inner_spec r checked before scratch _ finish.place rfl rfl

def PrefixSpec (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : RealSource.Memory) (scratch : BlockState)
    (result : CheckedHost.Returned) : Prop :=
  (∃ place : PrefixPlace r,result.objectId=place.objectId ∧ result.tensor=place.tensor) ∧
  (r.nonempty → DirectCumulativeSequence.Runs (RealSource.prefixLayout r BS) (r.B*r.H) "s" "o" Bool.false .fp32 .fp32
    (r.config BS).plans {RealSource.prefixInput r before scratch with undef:=fun _ _=>0}
    (RealSource.prefixResult r checked BS selected before scratch)) ∧
  (∀ i,i<prefixCount r → RealPhysical.prefixResult r checked BS selected before scratch result.tensor (result.tensor.byteBase+4*i)=
    RealSource.Prefix r before i) ∧
  (∀ address,Outside result.tensor.byteBase (prefixCount r) address →
    RealPhysical.prefixResult r checked BS selected before scratch result.tensor address=before address)

theorem prefix_spec (r : Admission.Prefix) (checked : r.check=Bool.true) (BS : Nat)
    (selected : BS=16 ∨ BS=32 ∨ BS=64) (before : RealSource.Memory) (scratch : BlockState)
    (result : CheckedHost.Returned) (place : PrefixPlace r)
    (identity : result.objectId=place.objectId) (tensor : result.tensor=place.tensor) :
    PrefixSpec r checked BS selected before scratch result := by
  refine ⟨⟨place,identity,tensor⟩,?_,RealPhysical.prefix_value r checked BS selected before scratch result.tensor,
    RealPhysical.prefix_frame r checked BS selected before scratch result.tensor⟩
  intro nonempty
  have source:=(RealSource.prefix_exists r checked BS selected before scratch).choose_spec.1
  have he : (r.config BS).empty=Bool.false := by simpa [HostFP.Cumulative.empty,Admission.Prefix.config,Admission.Prefix.nonempty] using nonempty
  rw [HostFP.Cumulative.plans,he]
  exact source

theorem prefix_returned (I : ScalarNative.Primitives) (r : Admission.Prefix) (checked : r.check=Bool.true)
    (bytes : ByteMemory.Memory) (rt : CheckedHost.PrefixRuntime r bytes) (fpScratch : State NativeGraph.Expr)
    (caller : Nat) (result : CheckedHost.Returned)
    (success : CheckedHost.runPrefix I r bytes rt fpScratch caller=.ok result)
    (before : RealSource.Memory) (scratch : BlockState) : PrefixSpec r checked rt.BS rt.selected before scratch result := by
  have post:=CheckedHost.prefix_complete I r bytes rt fpScratch caller
  rw [success] at post
  cases post with
  | empty _ _ place _=>exact prefix_spec r checked rt.BS rt.selected before scratch _ place rfl rfl
  | active _ _ _ _ _ _ finish _=>exact prefix_spec r checked rt.BS rt.selected before scratch _ finish.place rfl rfl

abbrev checked_inner := @CheckedHost.inner_complete
abbrev checked_prefix := @CheckedHost.prefix_complete
abbrev full_inner_source := @RealSource.inner_exists
abbrev full_prefix_source := @RealSource.prefix_exists
#axiomsClean inner_returned
#axiomsClean prefix_returned
#axiomsClean checked_inner
#axiomsClean checked_prefix
#axiomsClean full_inner_source
#axiomsClean full_prefix_source
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Correct
