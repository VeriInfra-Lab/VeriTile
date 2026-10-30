import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.HelperHost
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.SourceExecution
namespace VeriTile.Bench.Repairs.MatmulDequantInt4
open Helpers
noncomputable section

structure DequantRealClaim (s : Shape) (initial result : State ℝ) : Prop where
  values : ∀ k n,k<s.k → n<s.n → result.values .dequant (k*s.n+n)=dequantizedReal s initial k n
  frame : ∀ r a,r≠.dequant ∨ s.k*s.n≤a → result.values r a=initial.values r a
  packed : result.packed=initial.packed
theorem dequant_real (s : Shape) (g : Grid s.k s.n) (order : List Work) (complete : g.Complete order)
    (initial : State ℝ) : DequantRealClaim s initial (dequantRun s realDequant order initial) := by
  refine ⟨standalone_whole_real s g order complete initial,dequant_frame s realDequant g order complete initial,?_⟩
  rw [dequant_run_writes]
  exact (Writes.run_readonly .dequant s.n _ order initial initial (Writes.reflexive _ _)).2

structure FusedRealClaim (s : Shape) (initial result : State ℝ) : Prop where
  values : ∀ r n,r<s.m → n<s.n → result.values .output (r*s.n+n)=mathematical s initial r n
  frame : ∀ r a,r≠.output ∨ s.m*s.n≤a → result.values r a=initial.values r a
  packed : result.packed=initial.packed
theorem fused_real (s : Shape) (ng : Bool) (domain : Source.FusedRawScope s ng)
    (config : Source.FusedConfig) (member : config∈Source.fusedConfigs) (fusion : Nat → Nat → Fusion)
    (g : Grid s.m s.n) (order : List Work) (complete : g.Complete order) (initial : State ℝ) :
    FusedRealClaim s initial (fusedRun s config.tile realFused ng fusion order initial) := by
  refine ⟨?_,fused_frame s config.tile realFused ng fusion g order complete initial,?_⟩
  · intro r n hr hn
    rw [fused_complete s config.tile realFused ng fusion g order complete initial r n hr hn]
    exact fused_value_real s config.tile ng (Source.all_fused_domains s ng domain config member) fusion initial ⟨r,n⟩ hr
  · rw [fused_run_writes]
    exact (Writes.run_readonly .output s.n _ order initial initial (Writes.reflexive _ _)).2

structure S1RealClaim (s : Shape) (initial result : State ℝ) : Prop where
  values : ∀ r n,r<s.m → n<s.n → result.values .output (r*s.n+n)=mathematical s initial r n
  callerFrame : ∀ r a,r≠.output → r≠.dequant → result.values r a=initial.values r a
theorem s1_real (s : Shape) (plan : Library.Selected s) (trials : List (List Work))
    (dg : Grid s.k s.n) (og : Grid s.m s.n) (ds os : List Work)
    (hd : dg.Complete ds) (ho : og.Complete os) (initial : State ℝ) :
    S1RealClaim s initial (S1.run s realDequant Library.realOps plan ds os
      (Pipeline.prepared s 1 realDequant trials initial)) := by
  constructor
  · intro r n hr hn
    rw [S1.whole s realDequant Library.realOps plan dg og ds os hd ho _ r n hr hn,
      Pipeline.reference_readonly s realDequant Library.realOps plan initial _
        (Pipeline.prepared_readonly s 1 realDequant trials initial) r n,
      S1.reference_real s plan initial r n hr hn]
  · intro r a hr hd'
    rw [S1.frame s realDequant Library.realOps plan dg og ds os hd ho _ r a (Or.inl hr) (Or.inl hd')]
    exact (Pipeline.prepared_readonly s 1 realDequant trials initial).1 r a hd'

structure QuantRealClaim (s : Shape) (order : ReductionOrder s.group) (initial : State ℝ)
    (result : Quantized ℝ) : Prop where
  shape : result.weightShape=(s.k/8,s.n) ∧ result.scaleShape=(s.k/s.group,s.n) ∧
    result.zeroShape=(s.k/s.group,s.n/8) ∧ result.groupSize=s.group
  weights : ∀ p n,p<s.k/8 → n<s.n → result.weights (p*s.n+n)=weightSpec s HelperReal.arithmetic order initial p n
  zeros : ∀ g p,g<s.k/s.group → p<s.n/8 → result.zeros (g*(s.n/8)+p)=zeroSpec s HelperReal.arithmetic order initial g p
  scales : ∀ g n,g<s.k/s.group → n<s.n → ∀ hi lo,
    HelperReal.GroupRange (fun i : Fin s.group => input s initial g n i) hi lo →
    result.scales (g*s.n+n)=(hi-lo)/15
  zeroFormula : ∀ g n hi lo,HelperReal.GroupRange (fun i : Fin s.group => input s initial g n i) hi lo →
    zeroNibble s HelperReal.arithmetic order initial g n=HelperReal.mathZero hi lo
  weightFormula : ∀ k n hi lo,HelperReal.GroupRange (fun i : Fin s.group => input s initial (k/s.group) n i) hi lo →
    weightNibble s HelperReal.arithmetic order initial k n=HelperReal.mathWeight (initial.values .weight (k*s.n+n)) hi lo
theorem quantize_real (s : Shape) (hs : s.Valid) (order : ReductionOrder s.group) (initial : State ℝ) :
    QuantRealClaim s order initial (quantize s HelperReal.arithmetic order initial) := by
  refine ⟨⟨rfl,rfl,rfl,rfl⟩,quantize_result s hs _ order initial,?_,?_,?_,?_⟩
  · intro g p _ hp;exact quantize_zeros s hs _ order initial g p hp
  · intro g n _ hn hi lo range
    rw [quantize_scales s hs _ order initial g n hn]
    exact HelperReal.scale_independent s order initial g n hi lo range
  · exact HelperReal.zero_independent s order initial
  · exact HelperReal.weight_independent s order initial

/-- The standalone raw kernel and its host wrapper share the dequant component.
The actual s1 route composes dequantization with a complete selected library plan.
The unused fused raw kernel is proved separately; no fused host route is invented. -/
structure WholeReal : Prop where
  dequantizer : ∀ s (g : Grid s.k s.n) order,g.Complete order → ∀ initial,
    DequantRealClaim s initial (dequantRun s realDequant order initial)
  fused : ∀ s ng,Source.FusedRawScope s ng → ∀ c,c∈Source.fusedConfigs →
    ∀ fusion (g : Grid s.m s.n) order,g.Complete order → ∀ initial,
    FusedRealClaim s initial (fusedRun s c.tile realFused ng fusion order initial)
  s1 : ∀ s plan trials (dg : Grid s.k s.n) (og : Grid s.m s.n) ds os,
    dg.Complete ds → og.Complete os → ∀ initial,
    S1RealClaim s initial (S1.run s realDequant Library.realOps plan ds os (Pipeline.prepared s 1 realDequant trials initial))
  quantizer : ∀ s,s.Valid → ∀ order initial,QuantRealClaim s order initial (quantize s HelperReal.arithmetic order initial)
theorem whole_real_apis : WholeReal := ⟨dequant_real,fused_real,s1_real,quantize_real⟩

#axiomsClean dequant_real
#axiomsClean fused_real
#axiomsClean s1_real
#axiomsClean quantize_real
#axiomsClean whole_real_apis
end
end VeriTile.Bench.Repairs.MatmulDequantInt4
