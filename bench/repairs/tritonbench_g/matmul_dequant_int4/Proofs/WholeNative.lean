import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.WholeReal
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.FusedNative
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.MachineDomain
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
open Host Helpers Physical HelperBytes HelperHost
noncomputable section
set_option maxHeartbeats 5000000

structure DequantAllocation (x : PackedArguments) (dtype : DType) where
  tensor : Tensor
  layout : tensor.FreshLayout dtype x.shape.k x.shape.n x.weight.device
  independent : ∀ t∈[x.weight,x.scale,x.zero],Disjoint tensor t
def dequantHost (codec : Codec α) (x : PackedArguments) (dtype : DType) (allocation : DequantAllocation x dtype)
    (one : α) (p : DequantOps α) (trials : List (List Work)) (order : List Work) (initial : Bytes) (caller : Nat) : Host.Return :=
  ⟨allocation.tensor,Pipeline.dequantBytes codec x.shape one p trials order initial,x.weight.device,caller⟩
theorem dequant_caller_frame (codec : Codec α) (s : Shape) (one : α) (p : DequantOps α)
    (trials : List (List Work)) (order : List Work) (initial : Bytes) (r : Allocation) (b : Nat)
    (outside : r≠.value .dequant) : Pipeline.dequantBytes codec s one p trials order initial r b=initial r b := by
  unfold Pipeline.dequantBytes
  rw [Physical.frame codec .dequant s.n _ order _ r b (fun _ _ h => outside h.1)]
  unfold Pipeline.preparedBytes
  rw [Pipeline.trials_frame codec s p trials _ r b outside]
  simp [Pipeline.onesBytes,outside]
structure DequantNativeClaim (codec : Codec α) (x : PackedArguments) (dtype : DType)
    (a : DequantAllocation x dtype) (reference : DequantOps α) (initial : Bytes) (caller : Nat) (result : Host.Return) : Prop where
  identity : result.tensor=a.tensor
  layout : result.tensor.FreshLayout dtype x.shape.k x.shape.n x.weight.device
  device : result.callerDevice=caller ∧ result.launchDevice=x.weight.device
  bytes : ∀ k n,k<x.shape.k → n<x.shape.n → ∀ i : Fin codec.width,
    result.data (.value .dequant) (codec.width*(k*x.shape.n+n)+i.val)=
      codec.encode (dequantValue x.shape reference (load codec initial) ⟨k,n⟩) i
  frame : ∀ r b,r≠.value .dequant → result.data r b=initial r b
  indices : ∀ k n,k<Source.paddedK x.shape → n<Source.paddedN x.shape →
    Strides.qAddress x.strides k n<2^31 ∧ Strides.sAddress x.strides x.shape k n<2^31 ∧
    Strides.zAddress x.strides x.shape k n<2^31 ∧ k*x.shape.n+n<2^31
theorem dequant_native (codec : Codec α) (x : PackedArguments) (dtype : DType) (valid : PackedContract x dtype)
    (a : DequantAllocation x dtype) (one : α) (p : DequantOps α) (trials : List (List Work))
    (g : Grid x.shape.k x.shape.n) (order : List Work) (complete : g.Complete order) (initial : Bytes) (caller : Nat) :
    DequantNativeClaim codec x dtype a p initial caller (dequantHost codec x dtype a one p trials order initial caller) :=
  ⟨rfl,a.layout,⟨rfl,rfl⟩,Pipeline.dequant_result codec x.shape one p trials g order complete initial,
    dequant_caller_frame codec x.shape one p trials order initial,
    Strides.formed x.strides x.shape valid.indices⟩
theorem dequant_source (codec : Codec α) (x : PackedArguments) (dtype : DType) (valid : PackedContract x dtype)
    (a : DequantAllocation x dtype) (one : α) (p : DequantOps α) (trials : List (List Work))
    (g : Grid x.shape.k x.shape.n) (order : List Work) (complete : g.Complete order) (initial : Bytes) (caller : Nat) :
    load codec (dequantHost codec x dtype a one p trials order initial caller).data=
      SourceExecution.run x.shape x.strides p order (Pipeline.prepared x.shape one p trials (load codec initial)) := by
  change load codec (Pipeline.dequantBytes codec x.shape one p trials order initial)=_
  rw [Pipeline.dequant_load,SourceExecution.complete_run_eq x.shape valid.shape x.strides
    (packed_admitted x dtype valid) p g order complete]

structure S1NativeClaim (x : Arguments) (allocation : Allocations x) (reference : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (initial : Bytes) (caller : Nat) (result : Host.Return) : Prop where
  identity : result.tensor=x.output allocation.freshOutput
  layout : result.tensor.Ordinary .fp16 x.packed.shape.m x.packed.shape.n x.packed.weight.device
  device : result.callerDevice=caller ∧ result.launchDevice=x.packed.weight.device
  bytes : ∀ r n,r<x.packed.shape.m → n<x.packed.shape.n → ∀ i : Fin 2,
    result.data (.value .output) (2*(r*x.packed.shape.n+n)+i.val)=
      encode16 (S1.reference x.packed.shape (Native.dequant16 reference) (Native.library reference) plan
        (load halfCodec initial) r n) i
  frame : ∀ r b,r≠.value .output → r≠.value .dequant → result.data r b=initial r b
  indices : Strides.Guard x.packed.strides x.packed.shape
theorem s1_native (x : Arguments) (valid : Contract x) (a : Allocations x) (p : Native.Instructions)
    (plan : Library.Selected x.packed.shape) (trials : List (List Work))
    (dg : Grid x.packed.shape.k x.packed.shape.n) (og : Grid x.packed.shape.m x.packed.shape.n)
    (ds os : List Work) (hd : dg.Complete ds) (ho : og.Complete os) (initial : Bytes) (caller : Nat) :
    S1NativeClaim x a p plan initial caller (Host.call x a p plan trials ds os initial caller) := by
  refine ⟨rfl,return_geometry x valid a,⟨rfl,rfl⟩,
    returned_bytes x a p plan trials dg og ds os hd ho initial caller,?_,valid.packed.indices⟩
  intro r b ho' hd'
  exact Pipeline.caller_frame halfCodec x.packed.shape 0x3c00 (Native.dequant16 p) (Native.library p)
    plan trials ds os initial r b hd' ho'

structure FusedNativeClaim (s : Shape) (t : Tile) (reference : Native.Instructions) (ng : Bool)
    (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (initial result : Bytes) : Prop where
  bytes : ∀ r n,r<s.m → n<s.n → ∀ i : Fin 2,
    result (.value .output) (2*(r*s.n+n)+i.val)=
      encode16 (fusedValue s t (Native.fused reference kind) ng fusion (load halfCodec initial) ⟨r,n⟩) i
  frame : ∀ r b,r≠.value .output ∨ s.m*s.n≤b/2 → result r b=initial r b
  grid : blocksM s t*blocksN s t<2^31
  formed : ∀ r n k,r<ceil s.m t.bm*t.bm → n<s.n → k<s.k+t.bk →
    aAddress s r k<2^31 ∧ qAddress s k n<2^31 ∧ sAddress s k n<2^31 ∧
    zAddress s k n<2^31 ∧ cAddress s r n<2^31 ∧ cAddress s k n<2^31
theorem fused_native (s : Shape) (p : Native.Instructions) (ng : Bool) (domain : MachineDomain.Fused s ng)
    (c : Source.FusedConfig) (hc : c∈Source.fusedConfigs) (ts : List FusedNative.Trial)
    (active : ∀ trial∈ts,∀ w∈trial.2.order,w.Active s.m s.n)
    (kind : Library.MMAKind) (fusion : Nat → Nat → Fusion) (g : Grid s.m s.n)
    (order : List Work) (complete : g.Complete order) (initial : Bytes) :
    FusedNativeClaim s c.tile p ng kind fusion initial (FusedNative.call s p ng ts c.tile kind fusion order initial) :=
  ⟨FusedNative.result s p ng ts c.tile kind fusion g order complete initial,
    FusedNative.frame_all s p ng ts active c.tile kind fusion g order complete initial,
    MachineDomain.fused_grid_indices s ng domain c hc,MachineDomain.fused_formed s ng domain c hc⟩

structure QuantNativeClaim (codec : Codec α) (s : Shape) (dtype : DType) (reference : Arithmetic α)
    (order : ReductionOrder s.group) (initial : Bytes) (input : Tensor) (caller : Nat)
    (a : QuantAllocation s dtype input) (result : QuantReturn) : Prop where
  identity : result.weight=a.weight ∧ result.scale=a.scale ∧ result.zero=a.zero
  layout : result.weight.QuantLayout .int32 (s.k/8) s.n input.device ∧
    result.scale.QuantLayout dtype (s.k/s.group) s.n input.device ∧
    result.zero.QuantLayout .int32 (s.k/s.group) (s.n/8) input.device
  group : result.group=s.group
  device : result.callerDevice=caller
  callerFrame : result.data.caller=initial
  weights : ∀ p n,p<s.k/8 → n<s.n → ∀ i : Fin 4,
    result.data.weights (4*(p*s.n+n)+i.val)=encode32 (weightSpec s reference order (load codec initial) p n) i
  scales : ∀ g n,g<s.k/s.group → n<s.n → ∀ i : Fin codec.width,
    result.data.scales (codec.width*(g*s.n+n)+i.val)=codec.encode (scale s reference order (load codec initial) g n) i
  zeros : ∀ g p,g<s.k/s.group → p<s.n/8 → ∀ i : Fin 4,
    result.data.zeros (4*(g*(s.n/8)+p)+i.val)=encode32 (zeroSpec s reference order (load codec initial) g p) i
  independent : Independent [result.weight,result.scale,result.zero]
theorem quant16_native (s : Shape) (hs : s.Valid) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (input : Tensor) (caller : Nat) (a : QuantAllocation s .fp16 input) :
    QuantNativeClaim halfCodec s .fp16 (Native.quant16 p) order initial input caller a
      (quant16Call s p order initial scratch input a caller) :=
  ⟨⟨rfl,rfl,rfl⟩,⟨a.weightLayout,a.scaleLayout,a.zeroLayout⟩,rfl,rfl,rfl,
    weight_bytes halfCodec s hs _ order initial scratch,scale_bytes halfCodec s hs _ order initial scratch,
    zero_bytes halfCodec s hs _ order initial scratch,a.independent⟩
theorem quant32_native (s : Shape) (hs : s.Valid) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (input : Tensor) (caller : Nat) (a : QuantAllocation s .fp32 input) :
    QuantNativeClaim wordCodec s .fp32 (Native.quant32 p) order initial input caller a
      (quant32Call s p order initial scratch input a caller) :=
  ⟨⟨rfl,rfl,rfl⟩,⟨a.weightLayout,a.scaleLayout,a.zeroLayout⟩,rfl,rfl,rfl,
    weight_bytes wordCodec s hs _ order initial scratch,scale_bytes wordCodec s hs _ order initial scratch,
    zero_bytes wordCodec s hs _ order initial scratch,a.independent⟩

/-- Native public surface covers both standalone storage profiles, the actual
FP16 s1 wrapper, the unused fused raw kernel, and both quantizer input profiles.
Compiler fusion maps and torch.mm plan selection are metadata/ABI boundaries;
none supplies a numerical dot or completed output. -/
structure WholeNative (actual reference : Native.Instructions) : Prop where
  dequant16 : ∀ x,PackedContract x .fp16 → ∀ (a : DequantAllocation x .fp16) trials (g : Grid x.shape.k x.shape.n) order,
    g.Complete order → ∀ initial caller,
    DequantNativeClaim halfCodec x .fp16 a (Native.dequant16 reference) initial caller
      (dequantHost halfCodec x .fp16 a 0x3c00 (Native.dequant16 actual) trials order initial caller)
  dequant32 : ∀ x,PackedContract x .fp32 → ∀ (a : DequantAllocation x .fp32) trials (g : Grid x.shape.k x.shape.n) order,
    g.Complete order → ∀ initial caller,
    DequantNativeClaim wordCodec x .fp32 a (Native.dequant32 reference) initial caller
      (dequantHost wordCodec x .fp32 a 0x3f800000 (Native.dequant32 actual) trials order initial caller)
  s1 : ∀ x,Contract x → ∀ (a : Allocations x) plan trials
    (dg : Grid x.packed.shape.k x.packed.shape.n) (og : Grid x.packed.shape.m x.packed.shape.n) ds os,
    dg.Complete ds → og.Complete os → ∀ initial caller,
    S1NativeClaim x a reference plan initial caller (Host.call x a actual plan trials ds os initial caller)
  fused : ∀ s ng,MachineDomain.Fused s ng → ∀ c,c∈Source.fusedConfigs → ∀ ts,
    (∀ trial∈ts,(∃ config∈Source.fusedConfigs,trial.2.tile=config.tile) ∧
      ∀ w∈trial.2.order,w.Active s.m s.n) →
    ∀ kind fusion (g : Grid s.m s.n) order,g.Complete order → ∀ initial,
    FusedNativeClaim s c.tile reference ng kind fusion initial (FusedNative.call s actual ng ts c.tile kind fusion order initial)
  quant16 : ∀ s order initial weight rank,QuantDomain16 s actual order initial weight rank →
    ∀ scratch caller (a : QuantAllocation s .fp16 weight),
    QuantNativeClaim halfCodec s .fp16 (Native.quant16 reference) order initial weight caller a
      (quant16Call s actual order initial scratch weight a caller)
  quant32 : ∀ s order initial weight rank,QuantDomain32 s actual order initial weight rank →
    ∀ scratch caller (a : QuantAllocation s .fp32 weight),
    QuantNativeClaim wordCodec s .fp32 (Native.quant32 reference) order initial weight caller a
      (quant32Call s actual order initial scratch weight a caller)
theorem whole_native (p : Native.Instructions) : WholeNative p p := by
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  · intro x h a trials g order complete initial caller
    exact dequant_native halfCodec x .fp16 h a 0x3c00 (Native.dequant16 p) trials g order complete initial caller
  · intro x h a trials g order complete initial caller
    exact dequant_native wordCodec x .fp32 h a 0x3f800000 (Native.dequant32 p) trials g order complete initial caller
  · intro x h a plan trials dg og ds os hd ho initial caller
    exact s1_native x h a p plan trials dg og ds os hd ho initial caller
  · intro s ng domain c member ts active kind fusion g order complete initial
    exact fused_native s p ng domain c member ts (fun trial ht => (active trial ht).2) kind fusion g order complete initial
  · intro s order initial weight rank domain scratch caller a
    exact quant16_native s domain.shape p order initial scratch weight caller a
  · intro s order initial weight rank domain scratch caller a
    exact quant32_native s domain.shape p order initial scratch weight caller a
/-- Exactly 27 correspondence fields: 26 scalar families and one fixed K16 ISA
family. No whole-dot/output/source-execution or memory-result equality premise. -/
theorem whole_fp (actual reference : Native.Instructions) (numeric : Native.Correspondence actual reference) :
    WholeNative actual reference := by
  rw [←Native.correspondence_eq actual reference numeric]
  exact whole_native actual

#axiomsClean dequant_native
#axiomsClean dequant_caller_frame
#axiomsClean dequant_source
#axiomsClean s1_native
#axiomsClean fused_native
#axiomsClean quant16_native
#axiomsClean quant32_native
#axiomsClean whole_native
#axiomsClean whole_fp
end
end VeriTile.Bench.Repairs.MatmulDequantInt4
