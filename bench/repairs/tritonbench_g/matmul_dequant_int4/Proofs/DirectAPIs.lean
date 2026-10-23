import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectFused

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Physical Helpers HelperHost
noncomputable section
open Classical
set_option maxHeartbeats 5000000

/-- Derived source/address facts, including actual runtime singleton strides.
No caller-supplied coverage or source/output equality is accepted. -/
structure LaunchFacts (x : PackedArguments) : Prop where
  formed : ∀ k n,k<Source.paddedK x.shape → n<Source.paddedN x.shape →
    Strides.qAddress x.strides k n<2^31 ∧ Strides.sAddress x.strides x.shape k n<2^31 ∧
    Strides.zAddress x.strides x.shape k n<2^31 ∧ k*x.shape.n+n<2^31
  grids : ∀ choice : Fin 8,ceil x.shape.k (dequantConfig choice).bk*ceil x.shape.n (dequantConfig choice).bn<2^31
  source : ∀ (α : Type) (ops : DequantOps α) choice (initial : State α),
    SourceExecution.run x.shape x.strides ops (dequantOrder x.shape choice) initial=
      dequantRun x.shape ops (dequantOrder x.shape choice) initial

theorem launchFacts (x : PackedArguments) (dtype : Host.DType) (h : PackedContract x dtype) : LaunchFacts x := by
  refine ⟨Strides.formed _ _ h.indices,?_,?_⟩
  · intro choice
    exact MachineDomain.standalone_grid_indices _ _ h.indices _ (dequantConfig_mem choice)
  · intro α ops choice initial
    exact SourceExecution.complete_run_eq _ h.shape _ (packed_admitted _ _ h) ops
      (dequantGrid _ choice) _ (dequant_complete _ choice) initial

theorem dequant16_returned (p : PackedInfo) (device : Option Nat) (a : DequantAllocation (packed p 1) .fp16)
    (I : Native.Instructions) (choices : List (Fin 8)) (chosen : Fin 8) (heap : Heap) (caller : Nat)
    (result : TensorResult) (returned : dequantWrapper16 p device a I choices chosen heap caller=.ok result) :
    DequantClaim halfCodec (packed p 1) .fp16 a (Native.dequant16 I) heap caller result ∧ LaunchFacts (packed p 1) ∧ PackedAccepted p 1 .fp16 := by
  cases hd : checkDequant p .fp16 device with
  | error e => simp [dequantWrapper16,hd] at returned
  | ok unit =>
    have checked : checkDequant p .fp16 device=.ok () := by simpa using hd
    cases ha : checkAllocated a.tensor .fp16 with
    | error e => simp [dequantWrapper16,hd,ha] at returned
    | ok unit =>
      simp [dequantWrapper16,hd,ha] at returned
      subst result
      exact ⟨dequant_global halfCodec _ .fp16 rfl a _ _ choices chosen heap caller,
        launchFacts _ _ ((checkDequant_iff p .fp16 device).mp checked).1.2.1,
        ((checkDequant_iff p .fp16 device).mp checked).1⟩
theorem dequant32_returned (p : PackedInfo) (device : Option Nat) (a : DequantAllocation (packed p 1) .fp32)
    (I : Native.Instructions) (choices : List (Fin 8)) (chosen : Fin 8) (heap : Heap) (caller : Nat)
    (result : TensorResult) (returned : dequantWrapper32 p device a I choices chosen heap caller=.ok result) :
    DequantClaim wordCodec (packed p 1) .fp32 a (Native.dequant32 I) heap caller result ∧ LaunchFacts (packed p 1) ∧ PackedAccepted p 1 .fp32 := by
  cases hd : checkDequant p .fp32 device with
  | error e => simp [dequantWrapper32,hd] at returned
  | ok unit =>
    have checked : checkDequant p .fp32 device=.ok () := by simpa using hd
    cases ha : checkAllocated a.tensor .fp32 with
    | error e => simp [dequantWrapper32,hd,ha] at returned
    | ok unit =>
      simp [dequantWrapper32,hd,ha] at returned
      subst result
      exact ⟨dequant_global wordCodec _ .fp32 rfl a _ _ choices chosen heap caller,
        launchFacts _ _ ((checkDequant_iff p .fp32 device).mp checked).1.2.1,
        ((checkDequant_iff p .fp32 device).mp checked).1⟩
theorem s1_returned (p : S1Info) (a : Host.Allocations (arguments p)) (I : Native.Instructions)
    (plan : Library.Selected (arguments p).packed.shape) (choices : List (Fin 8)) (chosen : Fin 8)
    (heap : Heap) (caller : Nat) (result : TensorResult)
    (returned : s1Wrapper p a I plan choices chosen heap caller=.ok result) :
    S1Claim (arguments p) a I plan heap caller result ∧ LaunchFacts (arguments p).packed ∧ S1Accepted p := by
  cases hc : checkS1 p with
  | error e => simp [s1Wrapper,hc] at returned
  | ok unit =>
    have checked : checkS1 p=.ok () := by simpa using hc
    cases ha : checkS1Allocated p a with
    | error e => simp [s1Wrapper,hc,ha] at returned
    | ok unit =>
      simp [s1Wrapper,hc,ha] at returned
      subst result
      have h := ((checkS1_iff p).mp checked).2.1
      exact ⟨s1_global _ h a I plan choices chosen heap caller,launchFacts _ _ h.packed,(checkS1_iff p).mp checked⟩

theorem quant16_returned (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (a : QuantAllocation (quantShape p) .fp16 p.input.tensor) (heap : Heap) (caller : Nat) (result : QuantResult)
    (returned : quantWrapper16 p I order a heap caller=.ok result) :
    QuantClaim halfCodec (quantShape p) .fp16 (Native.quant16 I) order p.input.tensor a heap caller result ∧
    QuantMetadata p .fp16 ∧ QuantDomain16 (quantShape p) I order (bind (quantBase p.input.tensor) heap) p.input.tensor p.rank := by
  cases hc : checkQuant16 p I order (bind (quantBase p.input.tensor) heap) with
  | error e => simp [quantWrapper16,hc] at returned
  | ok unit =>
    have checked : checkQuant16 p I order (bind (quantBase p.input.tensor) heap)=.ok () := by simpa using hc
    have h := quant16_checked p I order _ checked
    simp [quantWrapper16,hc] at returned
    subst result
    exact ⟨quant_global halfCodec _ h.2.shape .fp16 rfl _ order _ a heap caller,h⟩
theorem quant32_returned (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (a : QuantAllocation (quantShape p) .fp32 p.input.tensor) (heap : Heap) (caller : Nat) (result : QuantResult)
    (returned : quantWrapper32 p I order a heap caller=.ok result) :
    QuantClaim wordCodec (quantShape p) .fp32 (Native.quant32 I) order p.input.tensor a heap caller result ∧
    QuantMetadata p .fp32 ∧ QuantDomain32 (quantShape p) I order (bind (quantBase p.input.tensor) heap) p.input.tensor p.rank := by
  cases hc : checkQuant32 p I order (bind (quantBase p.input.tensor) heap) with
  | error e => simp [quantWrapper32,hc] at returned
  | ok unit =>
    have checked : checkQuant32 p I order (bind (quantBase p.input.tensor) heap)=.ok () := by simpa using hc
    have h := quant32_checked p I order _ checked
    simp [quantWrapper32,hc] at returned
    subst result
    exact ⟨quant_global wordCodec _ h.2.shape .fp32 rfl _ order _ a heap caller,h⟩

structure CheckedNative (actual reference : Native.Instructions) : Prop where
  dequant16 : ∀ p device a choices chosen heap caller result,
    dequantWrapper16 p device a actual choices chosen heap caller=.ok result →
    DequantClaim halfCodec (packed p 1) .fp16 a (Native.dequant16 reference) heap caller result ∧ LaunchFacts (packed p 1) ∧ PackedAccepted p 1 .fp16
  dequant32 : ∀ p device a choices chosen heap caller result,
    dequantWrapper32 p device a actual choices chosen heap caller=.ok result →
    DequantClaim wordCodec (packed p 1) .fp32 a (Native.dequant32 reference) heap caller result ∧ LaunchFacts (packed p 1) ∧ PackedAccepted p 1 .fp32
  s1 : ∀ p a plan choices chosen heap caller result,s1Wrapper p a actual plan choices chosen heap caller=.ok result →
    S1Claim (arguments p) a reference plan heap caller result ∧ LaunchFacts (arguments p).packed ∧ S1Accepted p
  quant16 : ∀ p order a heap caller result,quantWrapper16 p actual order a heap caller=.ok result →
    QuantClaim halfCodec (quantShape p) .fp16 (Native.quant16 reference) order p.input.tensor a heap caller result ∧
    QuantMetadata p .fp16 ∧ QuantDomain16 (quantShape p) actual order (bind (quantBase p.input.tensor) heap) p.input.tensor p.rank
  quant32 : ∀ p order a heap caller result,quantWrapper32 p actual order a heap caller=.ok result →
    QuantClaim wordCodec (quantShape p) .fp32 (Native.quant32 reference) order p.input.tensor a heap caller result ∧
    QuantMetadata p .fp32 ∧ QuantDomain32 (quantShape p) actual order (bind (quantBase p.input.tensor) heap) p.input.tensor p.rank
  fusedRaw : ∀ p (h : RawAccepted p) choices chosen heap caller,
    RawClaim p reference chosen heap caller (rawCall p h actual choices chosen heap caller)

theorem checked_native (I : Native.Instructions) : CheckedNative I I :=
  ⟨fun p d a ts c h caller r ret => dequant16_returned p d a I ts c h caller r ret,
   fun p d a ts c h caller r ret => dequant32_returned p d a I ts c h caller r ret,
   fun p a plan ts c h caller r ret => s1_returned p a I plan ts c h caller r ret,
   fun p order a h caller r ret => quant16_returned p I order a h caller r ret,
   fun p order a h caller r ret => quant32_returned p I order a h caller r ret,
   fun p h ts c heap caller => raw_native p h I ts c heap caller⟩

theorem checked_if_local (actual reference : Native.Instructions)
    (numeric : Native.Correspondence actual reference) : CheckedNative actual reference := by
  rw [←Native.correspondence_eq actual reference numeric]
  exact checked_native actual

#axiomsClean checked_native
#axiomsClean checked_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
