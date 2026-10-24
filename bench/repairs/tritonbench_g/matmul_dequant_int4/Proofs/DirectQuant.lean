import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectNative

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Physical Helpers HelperHost HelperBytes
noncomputable section
open Classical
set_option maxHeartbeats 5000000

theorem disjoint_symm {a b : Host.Tensor} (h : Host.Disjoint a b) : Host.Disjoint b a := Or.symm h

structure QuantResult where
  weight : Host.Tensor
  scale : Host.Tensor
  zero : Host.Tensor
  group : Nat
  heap : Heap
  launchDevice : Nat
  callerDevice : Nat

def quantScratch (s : Shape) (dtype : Host.DType) (input : Host.Tensor)
    (a : QuantAllocation s dtype input) (heap : Heap) : Fin 3 → Nat → Byte :=
  fun i off => heap ((if i=0 then a.weight.base else if i=1 then a.scale.base else a.zero.base)+off)

def globalQuant (codec : Codec α) (s : Shape) (dtype : Host.DType) (ops : Arithmetic α)
    (order : ReductionOrder s.group) (input : Host.Tensor) (a : QuantAllocation s dtype input)
    (heap : Heap) (caller : Nat) : QuantResult :=
  let q := quantizeBytes codec s ops order (bind (quantBase input) heap) (quantScratch s dtype input a heap)
  ⟨a.weight,a.scale,a.zero,s.group,
    writeSpan a.zero.base (4*(s.k/s.group)*(s.n/8)) q.zeros
      (writeSpan a.scale.base (codec.width*(s.k/s.group)*s.n) q.scales
        (writeSpan a.weight.base (4*(s.k/8)*s.n) q.weights heap)),input.device,caller⟩

structure QuantClaim (codec : Codec α) (s : Shape) (dtype : Host.DType) (ops : Arithmetic α)
    (order : ReductionOrder s.group) (input : Host.Tensor) (a : QuantAllocation s dtype input)
    (heap : Heap) (caller : Nat) (result : QuantResult) : Prop where
  identity : result.weight=a.weight ∧ result.scale=a.scale ∧ result.zero=a.zero ∧ result.group=s.group
  metadata : result.weight.QuantLayout .int32 (s.k/8) s.n input.device ∧
    result.scale.QuantLayout dtype (s.k/s.group) s.n input.device ∧
    result.zero.QuantLayout .int32 (s.k/s.group) (s.n/8) input.device
  device : result.launchDevice=input.device ∧ result.callerDevice=caller
  weights : ∀ p n,p<s.k/8 → n<s.n → ∀ i : Fin 4,
    result.heap (result.weight.base+(4*(p*s.n+n)+i.val))=
      encode32 (weightSpec s ops order (load codec (bind (quantBase input) heap)) p n) i
  scales : ∀ g n,g<s.k/s.group → n<s.n → ∀ i : Fin codec.width,
    result.heap (result.scale.base+(codec.width*(g*s.n+n)+i.val))=
      codec.encode (scale s ops order (load codec (bind (quantBase input) heap)) g n) i
  zeros : ∀ g p,g<s.k/s.group → p<s.n/8 → ∀ i : Fin 4,
    result.heap (result.zero.base+(4*(g*(s.n/8)+p)+i.val))=
      encode32 (zeroSpec s ops order (load codec (bind (quantBase input) heap)) g p) i
  frame : ∀ address,
    ¬(a.weight.base≤address ∧ address<a.weight.base+4*(s.k/8)*s.n) →
    ¬(a.scale.base≤address ∧ address<a.scale.base+codec.width*(s.k/s.group)*s.n) →
    ¬(a.zero.base≤address ∧ address<a.zero.base+4*(s.k/s.group)*(s.n/8)) → result.heap address=heap address
  inputFrame : ∀ off,off < input.storageBytes → result.heap (input.base+off)=heap (input.base+off)
  independent : Host.Independent [result.weight,result.scale,result.zero]

theorem quant_global (codec : Codec α) (s : Shape) (hs : s.Valid) (dtype : Host.DType)
    (width : codec.width=dtype.width) (ops : Arithmetic α) (order : ReductionOrder s.group)
    (input : Host.Tensor) (a : QuantAllocation s dtype input) (heap : Heap) (caller : Nat) :
    QuantClaim codec s dtype ops order input a heap caller (globalQuant codec s dtype ops order input a heap caller) := by
  have qcap : 4*(s.k/8)*s.n≤a.weight.storageBytes := ordinary_capacity _ _ _ _ _ a.weightLayout.1
  have scap : codec.width*(s.k/s.group)*s.n≤a.scale.storageBytes := by
    rw [width]; exact ordinary_capacity _ _ _ _ _ a.scaleLayout.1
  have zcap : 4*(s.k/s.group)*(s.n/8)≤a.zero.storageBytes := ordinary_capacity _ _ _ _ _ a.zeroLayout.1
  have pairs := a.independent
  simp only [Host.Independent,List.pairwise_cons,List.mem_cons,List.mem_singleton,List.pairwise_singleton,and_true] at pairs
  have qs : Host.Disjoint a.weight a.scale := pairs.1 a.scale (Or.inl rfl)
  have qz : Host.Disjoint a.weight a.zero := pairs.1 a.zero (Or.inr (Or.inl rfl))
  have sz : Host.Disjoint a.scale a.zero := pairs.2.1 a.zero (Or.inl rfl)
  refine ⟨⟨rfl,rfl,rfl,rfl⟩,⟨a.weightLayout,a.scaleLayout,a.zeroLayout⟩,⟨rfl,rfl⟩,?_,?_,?_,?_,?_,a.independent⟩
  · intro p n hp hn i
    have coord := rowMajor_bound _ _ _ _ hp hn
    have ib := i.isLt
    have off : 4*(p*s.n+n)+i.val<4*(s.k/8)*s.n := by nlinarith
    dsimp only [globalQuant]
    change writeSpan _ _ _ _ _=_
    rw [writeSpan_disjoint _ _ (disjoint_symm qz) _ zcap _ _ _ (off.trans_le qcap),
      writeSpan_disjoint _ _ (disjoint_symm qs) _ scap _ _ _ (off.trans_le qcap),writeSpan_byte _ _ _ _ _ off]
    exact weight_bytes codec s hs ops order _ _ p n hp hn i
  · intro g n hg hn i
    have coord := rowMajor_bound _ _ _ _ hg hn
    have ib := i.isLt
    have off : codec.width*(g*s.n+n)+i.val<codec.width*(s.k/s.group)*s.n := by nlinarith
    dsimp only [globalQuant]
    change writeSpan _ _ _ _ _=_
    rw [writeSpan_disjoint _ _ (disjoint_symm sz) _ zcap _ _ _ (off.trans_le scap),writeSpan_byte _ _ _ _ _ off]
    exact scale_bytes codec s hs ops order _ _ g n hg hn i
  · intro g p hg hp i
    have coord := rowMajor_bound _ _ _ _ hg hp
    have ib := i.isLt
    have off : 4*(g*(s.n/8)+p)+i.val<4*(s.k/s.group)*(s.n/8) := by nlinarith
    dsimp only [globalQuant]
    change writeSpan _ _ _ _ _=_
    rw [writeSpan_byte _ _ _ _ _ off]
    exact zero_bytes codec s hs ops order _ _ g p hg hp i
  · intro address hq hs hz
    dsimp only [globalQuant]
    change writeSpan _ _ _ _ _=_
    rw [writeSpan_frame _ _ _ _ _ hz,writeSpan_frame _ _ _ _ _ hs,writeSpan_frame _ _ _ _ _ hq]
  · intro off bound
    have iq := a.callerDisjoint a.weight (by simp)
    have is := a.callerDisjoint a.scale (by simp)
    have iz := a.callerDisjoint a.zero (by simp)
    dsimp only [globalQuant]
    change writeSpan _ _ _ _ _=_
    rw [writeSpan_disjoint _ _ (disjoint_symm iz) _ zcap _ _ off bound,
      writeSpan_disjoint _ _ (disjoint_symm is) _ scap _ _ off bound,
      writeSpan_disjoint _ _ (disjoint_symm iq) _ qcap _ _ off bound]

def quantWrapper16 (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (a : QuantAllocation (quantShape p) .fp16 p.input.tensor) (heap : Heap) (caller : Nat) : Except Error QuantResult := do
  checkQuant16 p I order (bind (quantBase p.input.tensor) heap)
  pure (globalQuant halfCodec (quantShape p) .fp16 (Native.quant16 I) order p.input.tensor a heap caller)
def quantWrapper32 (p : QuantInfo) (I : Native.Instructions) (order : ReductionOrder (quantShape p).group)
    (a : QuantAllocation (quantShape p) .fp32 p.input.tensor) (heap : Heap) (caller : Nat) : Except Error QuantResult := do
  checkQuant32 p I order (bind (quantBase p.input.tensor) heap)
  pure (globalQuant wordCodec (quantShape p) .fp32 (Native.quant32 I) order p.input.tensor a heap caller)

#axiomsClean quant_global
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
