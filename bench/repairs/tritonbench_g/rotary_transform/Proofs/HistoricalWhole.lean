import bench.repairs.tritonbench_g.rotary_transform.Proofs.HistoricalPublic
namespace VeriTile.Bench.Repairs.RotaryTransform.Public
open Native Physical Host

/-- Ordinary checked ABI conditions over the initial caller allocation bytes.
Allocation separation is represented by distinct Region identities, with only
X/output aliasing selected by inplace. No result-valued field is accepted. -/
structure Admitted (call : Call) (views : InputViews) (initial : Bytes) : Prop where
  config : (callerConfiguration call.shape call.abi views initial).Valid
  indices : IndexABI (callerConfiguration call.shape call.abi views initial)
  viewStrides : 0<views.cosineRow ∧ 0<views.cosineColumn ∧ 0<views.sineRow ∧ 0<views.sineColumn ∧ 0<views.offsetsStride
  cumulativeStride : 0<call.abi.cumulativeStride
  fresh : call.shape.inplace=false → call.freshIdentity≠call.inputIdentity

structure NativeClaim (call : Call) (views : InputViews) (callerDevice : Nat)
    (actual target : Instructions) (initial : Bytes) (result : Returned Byte) : Prop where
  returned : result.valueIdentity=(if call.shape.inplace then call.inputIdentity else call.freshIdentity)
  region : result.valueRegion=call.shape.output
  shape : result.shape=call.shape
  dtype : result.fp32=true
  device : result.device=call.device ∧ result.deviceRestored=callerDevice
  source : load result.memory=Native.project (source (callerConfiguration call.shape call.abi views initial) (Native.ops actual)
    (Native.liftMemory (load (prepareBytes call.shape (normalizeFor call.shape call.abi
      (stageInputs call.shape views call.abi.offsets64 initial))))))
  bytes : ∀ b r h d,b<call.shape.batches → r<(callerConfiguration call.shape call.abi views initial).length b →
    h<call.shape.heads → d<call.shape.width → ∀ i : Fin 4,
    result.memory call.shape.output (4*call.shape.address ((callerConfiguration call.shape call.abi views initial).start b+r) h d+i.val)=
      encode (nativeExpected call views target initial b r h d) i
  frame : ∀ q b,CallerRegion q → (q≠call.shape.output ∨ size call.shape≤b/4) → result.memory q b=initial q b

theorem whole_native (call : Call) (views : InputViews) (initial : Bytes) (domain : Admitted call views initial)
    (callerDevice : Nat) (p : Instructions) :
    NativeClaim call views callerDevice p p initial (invoke call views callerDevice p initial) := by
  refine ⟨rfl,rfl,rfl,rfl,⟨rfl,rfl⟩,?_,?_,?_⟩
  · exact source_interpretation call views callerDevice p initial domain.config
  · exact payload call views callerDevice p initial domain.config
  · exact frame call views callerDevice p initial domain.config

theorem whole_fp (call : Call) (views : InputViews) (initial : Bytes) (domain : Admitted call views initial)
    (callerDevice : Nat) (actual target : Instructions) (ports : Obligations actual target) :
    NativeClaim call views callerDevice actual target initial (invoke call views callerDevice actual initial) := by
  rw [←instructions_eq actual target ports]
  exact whole_native call views initial domain callerDevice actual

/-- Every returned token belongs to a batch interval, even with empty packed
sequences. Thus the byte theorem covers the entire returned logical tensor. -/
theorem every_token (call : Call) (views : InputViews) (initial : Bytes) (domain : Admitted call views initial)
    (token : Nat) (ht : token<call.shape.total) :
    ∃ b r,b<call.shape.batches ∧ r<(callerConfiguration call.shape call.abi views initial).length b ∧
      (callerConfiguration call.shape call.abi views initial).start b+r=token := by
  let c := callerConfiguration call.shape call.abi views initial
  have bound : token<c.start c.shape.batches := by rw [start_total c domain.config];exact ht
  obtain ⟨b,hb,lo,hi⟩ := intervals_cover c domain.config c.shape.batches (by omega) token bound
  refine ⟨b,token-c.start b,hb,?_,?_⟩
  · change token-c.start b<c.length b
    omega
  · change c.start b+(token-c.start b)=token
    omega

structure RealClaim (call : Call) (views : InputViews) (callerDevice : Nat) (interpret : Word → ℝ)
    (initial : Bytes) (result : Returned ℝ) : Prop where
  returned : result.valueIdentity=(if call.shape.inplace then call.inputIdentity else call.freshIdentity)
  region : result.valueRegion=call.shape.output
  shape : result.shape=call.shape
  dtype : result.fp32=true
  device : result.device=call.device ∧ result.deviceRestored=callerDevice
  source : result.memory=source (callerConfiguration call.shape call.abi views initial) realOps
    (prepare call.shape (fun r a => interpret (load (stageInputs call.shape views call.abi.offsets64 initial) r a)))
  values : ∀ b r h d,b<call.shape.batches → r<(callerConfiguration call.shape call.abi views initial).length b →
    h<call.shape.heads → d<call.shape.width →
    result.memory call.shape.output (call.shape.address ((callerConfiguration call.shape call.abi views initial).start b+r) h d)=
      realExpected call views interpret initial b r h d
  frame : ∀ q a,CallerRegion q → (q≠call.shape.output ∨ size call.shape≤a) →
    result.memory q a=interpret (load initial q a)

theorem ideal_source (call : Call) (views : InputViews) (callerDevice : Nat) (interpret : Word → ℝ) (initial : Bytes)
    (hc : (callerConfiguration call.shape call.abi views initial).Valid) :
    (ideal call views callerDevice interpret initial).memory=
      source (callerConfiguration call.shape call.abi views initial) realOps
        (prepare call.shape (fun r a => interpret (load (stageInputs call.shape views call.abi.offsets64 initial) r a))) := by
  have hc' : (originalConfiguration call.shape call.abi (stageInputs call.shape views call.abi.offsets64 initial)).Valid := by
    rw [staged_configuration];exact hc
  unfold ideal
  rw [mathematical_memory _ _ _ _ hc',staged_configuration]

theorem ideal_frame (call : Call) (views : InputViews) (callerDevice : Nat) (interpret : Word → ℝ) (initial : Bytes)
    (hc : (callerConfiguration call.shape call.abi views initial).Valid)
    (q : Region) (a : Nat) (caller : CallerRegion q) (outside : q≠call.shape.output ∨ size call.shape≤a) :
    (ideal call views callerDevice interpret initial).memory q a=interpret (load initial q a) := by
  rw [ideal_source _ _ _ _ _ hc,source_eq_run _ hc,run_frame _ realOps _ q a (outside_writes _ hc q a outside)]
  rw [prepare,if_neg (outside_copy call.shape q a outside)]
  congr 1
  unfold load
  congr 1
  funext i
  exact stage_frame call.shape views call.abi.offsets64 initial q (4*a+i.val) caller

theorem whole_real (call : Call) (views : InputViews) (initial : Bytes) (domain : Admitted call views initial)
    (callerDevice : Nat) (interpret : Word → ℝ) :
    RealClaim call views callerDevice interpret initial (ideal call views callerDevice interpret initial) := by
  refine ⟨rfl,rfl,rfl,rfl,⟨rfl,rfl⟩,?_,?_,?_⟩
  · exact ideal_source call views callerDevice interpret initial domain.config
  · exact ideal_payload call views callerDevice interpret initial domain.config
  · exact ideal_frame call views callerDevice interpret initial domain.config

#axiomsClean whole_native
#axiomsClean whole_fp
#axiomsClean every_token
#axiomsClean whole_real
end VeriTile.Bench.Repairs.RotaryTransform.Public
