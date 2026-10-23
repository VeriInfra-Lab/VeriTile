import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectFused

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Helpers HelperHost
noncomputable section
open Classical
set_option maxHeartbeats 5000000

/-- Real payloads are arbitrary typed memory, independently quantified; no native
codec, representability, or finite floating-point premise restricts them. -/
structure RealTensorResult where
  tensor : Host.Tensor
  data : State ℝ
  launchDevice : Nat
  callerDevice : Nat

def realDequantCall (p : PackedInfo) (dtype : Host.DType) (a : DequantAllocation (packed p 1) dtype)
    (choices : List (Fin 8)) (chosen : Fin 8) (initial : State ℝ) (caller : Nat) : RealTensorResult :=
  let x := packed p 1
  ⟨a.tensor,SourceExecution.run x.shape x.strides realDequant (dequantOrder x.shape chosen)
    (Pipeline.prepared x.shape 1 realDequant (trialOrders x.shape choices) initial),x.weight.device,caller⟩

structure RealDequantClaim (p : PackedInfo) (dtype : Host.DType) (a : DequantAllocation (packed p 1) dtype)
    (initial : State ℝ) (caller : Nat) (result : RealTensorResult) : Prop where
  identity : result.tensor=a.tensor
  metadata : result.tensor.FreshLayout dtype (packed p 1).shape.k (packed p 1).shape.n p.weight.tensor.device
  device : result.launchDevice=p.weight.tensor.device ∧ result.callerDevice=caller
  whole : DequantRealClaim (packed p 1).shape initial result.data

theorem prepared_frame (s : Shape) (choices : List (Fin 8)) (initial : State ℝ)
    (r : Region) (off : Nat) (outside : r≠.dequant ∨ s.k*s.n≤off) :
    (Pipeline.prepared s 1 realDequant (trialOrders s choices) initial).values r off=initial.values r off := by
  unfold Pipeline.prepared
  rw [Autotune.dequant_frame _ _ _ (trial_active s choices) _ r off outside]
  rcases outside with hr|hb
  · simp [Pipeline.ones,hr]
  · simp [Pipeline.ones,show ¬off<s.k*s.n from Nat.not_lt.mpr hb]

theorem real_dequant (p : PackedInfo) (dtype : Host.DType) (device : Option Nat)
    (checked : checkDequant p dtype device=.ok ()) (a : DequantAllocation (packed p 1) dtype)
    (choices : List (Fin 8)) (chosen : Fin 8) (initial : State ℝ) (caller : Nat) :
    RealDequantClaim p dtype a initial caller (realDequantCall p dtype a choices chosen initial caller) := by
  have h := ((checkDequant_iff p dtype device).mp checked).1.2.1
  refine ⟨rfl,a.layout,⟨rfl,rfl⟩,?_⟩
  change DequantRealClaim _ initial (SourceExecution.run _ _ _ _ _)
  rw [SourceExecution.complete_run_eq _ h.shape _ (packed_admitted _ _ h) _
    (dequantGrid _ chosen) _ (dequant_complete _ chosen)]
  refine ⟨?_,?_,?_⟩
  · intro k n hk hn
    rw [VeriTile.Bench.Repairs.MatmulDequantInt4.dequant_complete _ realDequant (dequantGrid _ chosen) _
      (dequant_complete _ chosen) _ k n hk hn,
      dequant_stable _ realDequant initial _ (Pipeline.prepared_readonly _ 1 realDequant _ initial)]
    exact standalone_real_value _ initial k n
  · intro r off outside
    rw [dequant_frame _ realDequant (dequantGrid _ chosen) _ (dequant_complete _ chosen) _ r off outside]
    exact prepared_frame _ choices initial r off outside
  · rw [dequant_run_writes]
    exact (Writes.run_readonly .dequant _ _ _ initial _ (Pipeline.prepared_readonly _ 1 realDequant _ initial)).2

def realS1Call (p : S1Info) (a : Host.Allocations (arguments p)) (plan : Library.Selected (arguments p).packed.shape)
    (choices : List (Fin 8)) (chosen : Fin 8) (initial : State ℝ) (caller : Nat) : RealTensorResult :=
  let x := arguments p
  ⟨x.output a.freshOutput,
    Library.run x.packed.shape Library.realOps plan (outputOrder x.packed.shape)
      (SourceExecution.run x.packed.shape x.packed.strides realDequant (dequantOrder x.packed.shape chosen)
        (Pipeline.prepared x.packed.shape 1 realDequant (trialOrders x.packed.shape choices) initial)),
    x.packed.weight.device,caller⟩
structure RealS1Claim (p : S1Info) (a : Host.Allocations (arguments p)) (initial : State ℝ)
    (caller : Nat) (result : RealTensorResult) : Prop where
  identity : result.tensor=(arguments p).output a.freshOutput
  metadata : result.tensor.Ordinary .fp16 (arguments p).packed.shape.m (arguments p).packed.shape.n p.packedInfo.weight.tensor.device
  device : result.launchDevice=p.packedInfo.weight.tensor.device ∧ result.callerDevice=caller
  whole : S1RealClaim (arguments p).packed.shape initial result.data
  frame : ∀ r off,(r≠.output ∨ (arguments p).packed.shape.m*(arguments p).packed.shape.n≤off) →
    (r≠.dequant ∨ (arguments p).packed.shape.k*(arguments p).packed.shape.n≤off) →
    result.data.values r off=initial.values r off

theorem real_s1 (p : S1Info) (checked : checkS1 p=.ok ()) (a : Host.Allocations (arguments p))
    (plan : Library.Selected (arguments p).packed.shape) (choices : List (Fin 8)) (chosen : Fin 8)
    (initial : State ℝ) (caller : Nat) : RealS1Claim p a initial caller (realS1Call p a plan choices chosen initial caller) := by
  have h := ((checkS1_iff p).mp checked).2.1
  refine ⟨rfl,return_geometry _ h a,⟨rfl,rfl⟩,?_,?_⟩
  · change S1RealClaim _ initial (Library.run _ _ _ _ (SourceExecution.run _ _ _ _ _))
    rw [SourceExecution.complete_run_eq _ h.packed.shape _ (packed_admitted _ _ h.packed) _
      (dequantGrid _ chosen) _ (dequant_complete _ chosen)]
    exact s1_real _ plan (trialOrders _ choices) (dequantGrid _ chosen) (outputGrid _) _ _
      (dequant_complete _ chosen) (output_complete _) initial
  · intro r off ho hd
    change (Library.run _ _ _ _ (SourceExecution.run _ _ _ _ _)).values r off=_
    rw [SourceExecution.complete_run_eq _ h.packed.shape _ (packed_admitted _ _ h.packed) _
      (dequantGrid _ chosen) _ (dequant_complete _ chosen)]
    rw [Library.whole_frame _ Library.realOps plan (outputGrid _) _ (output_complete _) _ r off ho,
      dequant_frame _ realDequant (dequantGrid _ chosen) _ (dequant_complete _ chosen) _ r off hd]
    exact prepared_frame _ choices initial r off hd

structure RealQuantResult where
  weight : Host.Tensor
  scale : Host.Tensor
  zero : Host.Tensor
  group : Nat
  quantized : Quantized ℝ
  callerData : State ℝ
  launchDevice : Nat
  callerDevice : Nat

def realQuantCall (p : QuantInfo) (dtype : Host.DType) (a : QuantAllocation (quantShape p) dtype p.input.tensor)
    (order : ReductionOrder (quantShape p).group) (initial : State ℝ) (caller : Nat) : RealQuantResult :=
  ⟨a.weight,a.scale,a.zero,p.group,quantize (quantShape p) HelperReal.arithmetic order initial,
    initial,p.input.tensor.device,caller⟩
structure RealQuantClaim (p : QuantInfo) (dtype : Host.DType) (a : QuantAllocation (quantShape p) dtype p.input.tensor)
    (order : ReductionOrder (quantShape p).group) (initial : State ℝ) (caller : Nat) (result : RealQuantResult) : Prop where
  identity : result.weight=a.weight ∧ result.scale=a.scale ∧ result.zero=a.zero ∧ result.group=p.group
  metadata : result.weight.QuantLayout .int32 ((quantShape p).k/8) (quantShape p).n p.input.tensor.device ∧
    result.scale.QuantLayout dtype ((quantShape p).k/p.group) (quantShape p).n p.input.tensor.device ∧
    result.zero.QuantLayout .int32 ((quantShape p).k/p.group) ((quantShape p).n/8) p.input.tensor.device
  device : result.launchDevice=p.input.tensor.device ∧ result.callerDevice=caller
  whole : QuantRealClaim (quantShape p) order initial result.quantized
  frame : result.callerData=initial
  independent : Host.Independent [result.weight,result.scale,result.zero]
theorem real_quant (p : QuantInfo) (dtype : Host.DType) (h : QuantMetadata p dtype)
    (a : QuantAllocation (quantShape p) dtype p.input.tensor) (order : ReductionOrder (quantShape p).group)
    (initial : State ℝ) (caller : Nat) :
    RealQuantClaim p dtype a order initial caller (realQuantCall p dtype a order initial caller) :=
  ⟨⟨rfl,rfl,rfl,rfl⟩,⟨a.weightLayout,a.scaleLayout,a.zeroLayout⟩,⟨rfl,rfl⟩,
    quantize_real _ h.2.2.1 order initial,rfl,a.independent⟩

def rawRealCall (p : RawInfo) (h : RawAccepted p) (choices : List RawChoice) (chosen : RawChoice)
    (initial : State ℝ) (caller : Nat) : RealTensorResult :=
  let s := rawShape p
  ⟨p.output.tensor,fusedRun s (fusedConfig chosen.config).tile realFused p.noGroups chosen.fusion
    (rawOrder s h.2.1.shape chosen.config)
    (Autotune.fusedTrials s realFused p.noGroups ((rawTrials s h.2.1.shape choices).map Prod.snd) initial),
    p.input.tensor.device,caller⟩
theorem fused_trials_frame (s : Shape) (trials : List Autotune.FusedTrial) (ng : Bool)
    (active : ∀ trial∈trials,∀ w∈trial.order,w.Active s.m s.n) (initial : State ℝ)
    (r : Region) (off : Nat) (outside : r≠.output ∨ s.m*s.n≤off) :
    (Autotune.fusedTrials s realFused ng trials initial).values r off=initial.values r off := by
  induction trials generalizing initial with
  | nil => rfl
  | cons trial rest ih =>
    change (Autotune.fusedTrials s realFused ng rest (fusedRun _ _ _ _ _ _ _)).values r off=_
    rw [ih (fun t ht => active t (by simp [ht])),fused_run_writes]
    apply Writes.frame
    intro w hw ⟨hr,ha⟩
    have h := active trial (by simp) w hw
    have hb := rowMajor_bound _ _ _ _ h.1 h.2
    rcases outside with h|h
    · exact h hr
    · omega

structure RealRawClaim (p : RawInfo) (initial : State ℝ) (caller : Nat) (result : RealTensorResult) : Prop where
  identity : result.tensor=p.output.tensor
  metadata : result.tensor.Ordinary .fp16 (rawShape p).m (rawShape p).n p.input.tensor.device
  device : result.launchDevice=p.input.tensor.device ∧ result.callerDevice=caller
  values : ∀ r n,r<(rawShape p).m → n<(rawShape p).n →
    result.data.values .output (r*(rawShape p).n+n)=mathematical (rawShape p) initial r n
  inputs : Writes.Readonly .output initial result.data
  frame : ∀ r off,r≠.output ∨ (rawShape p).m*(rawShape p).n≤off → result.data.values r off=initial.values r off

theorem real_raw (p : RawInfo) (h : RawAccepted p) (choices : List RawChoice) (chosen : RawChoice)
    (initial : State ℝ) (caller : Nat) : RealRawClaim p initial caller (rawRealCall p h choices chosen initial caller) := by
  refine ⟨rfl,h.2.2.2.2.2.2.2.1.2.2.2.2.2.2.1,⟨rfl,rfl⟩,?_,?_,?_⟩
  · intro r n hr hn
    change (fusedRun _ _ _ _ _ _ _).values _ _=_
    rw [Autotune.fused_absorbed (rawShape p) realFused p.noGroups
      ((rawTrials (rawShape p) h.2.1.shape choices).map Prod.snd) (fusedConfig chosen.config).tile chosen.fusion
      (rawGrid _ h.2.1.shape chosen.config) (rawOrder _ h.2.1.shape chosen.config) (List.Perm.refl _) initial r n hr hn]
    exact fused_value_real _ _ _ (Source.all_fused_domains _ _ h.2.1 _ (fusedConfig_mem chosen.config)) _ initial ⟨r,n⟩ hr
  · change Writes.Readonly .output initial (fusedRun _ _ _ _ _ _ _)
    rw [fused_run_writes]
    exact Writes.run_readonly _ _ _ _ initial _
      (Autotune.fused_readonly _ realFused p.noGroups _ initial initial (Writes.reflexive _ _))
  · intro r off outside
    change (fusedRun _ _ _ _ _ _ _).values r off=_
    rw [fused_frame _ _ realFused p.noGroups chosen.fusion (rawGrid _ h.2.1.shape chosen.config)
      (rawOrder _ h.2.1.shape chosen.config) (List.Perm.refl _) _ r off outside]
    apply fused_trials_frame _ _ _ _ initial r off outside
    intro trial member w used
    obtain ⟨pair,hpair,rfl⟩ := List.mem_map.mp member
    exact raw_trials_active _ h.2.1.shape choices pair hpair w used

#axiomsClean real_dequant
#axiomsClean real_s1
#axiomsClean real_quant
#axiomsClean real_raw
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
