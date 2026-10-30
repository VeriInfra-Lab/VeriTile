import bench.repairs.tritonbench_g.chunk_retention.Proofs.RealMemory

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical BigOperators
set_option maxHeartbeats 3500000

structure RealForwardReturn where
  output : Tensor
  finalState : Option Tensor
  saved : List Tensor
  launches : List Phase
  launchDevice : Nat
  restoredDevice : Nat
  memory : RealMemory

structure RealBackwardReturn where
  dq : Tensor
  dk : Tensor
  dv : Tensor
  seedGradient : Option Tensor
  flagGradient : Option Tensor
  launches : List Phase
  launchDevice : Nat
  restoredDevice : Nat
  memory : RealMemory

def realForwardReturn (c : Shape) (f : ForwardObjects c) (scale : ℝ)
    (tree : SumTree) (priorDevice : Nat) (before : RealMemory) : RealForwardReturn :=
  ⟨f.output,f.finalState,[f.q,f.k,f.v,f.history],forwardLaunches c,f.q.device,priorDevice,
    forwardRealMemory c f scale tree before⟩

def realBackwardReturn (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (priorDevice : Nat) (later : RealMemory) : RealBackwardReturn :=
  ⟨b.dq,b.dk,b.dv,none,none,backwardLaunches c,f.q.device,priorDevice,backwardRealMemory c f b scale tree later⟩

def forwardRealCall (c : Shape) (f : ForwardObjects c) (scale : ℝ)
    (tree : SumTree) (priorDevice : Nat) (before : RealMemory) : Option RealForwardReturn :=
  if forwardChecks c f then some (realForwardReturn c f scale tree priorDevice before) else none

def backwardRealCall (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (priorDevice : Nat) (later : RealMemory) : Option RealBackwardReturn :=
  if backwardChecks c f b then some (realBackwardReturn c f b scale tree priorDevice later) else none

/-- Input-only retention recurrence at every actual returned output coordinate. -/
structure RealForwardValues (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (result : RealForwardReturn) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    result.memory (result.output.dataPointer+4*c.valueOffset h t j)=MathTarget.output c xs scale Bool.true h t j
  finalState : ∀ final,result.finalState=some final → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    result.memory (final.dataPointer+4*c.stateOffset h e j)=MathTarget.finalState c xs Bool.true h e j

/-- Input-only closed gradient sums at actual returned gradient pointers. -/
structure RealBackwardValues (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (result : RealBackwardReturn) : Prop where
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    result.memory (result.dq.dataPointer+4*c.keyOffset h t e)=MathTarget.queryGradient c xs scale Bool.true h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    result.memory (result.dk.dataPointer+4*c.keyOffset h t e)=MathTarget.keyGradient c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    result.memory (result.dv.dataPointer+4*c.valueOffset h t j)=MathTarget.valueGradient c xs scale h t j

theorem forward_real_values (c : Shape) (f : ForwardObjects c) (scale : ℝ) (tree : SumTree)
    (priorDevice : Nat) (before : RealMemory) (checks : forwardChecks c f=Bool.true)
    (fresh : SeparateStores f.tensor (Separate.forwardFootprint c f.flag)) :
    RealForwardValues c (inputsOfHeap (forwardRealHeap c f before)) scale (realForwardReturn c f scale tree priorDevice before) := by
  have admit:=forward_store_admission c f checks
  have source:=Separate.forward_real c scale tree f.flag (forwardRealHeap c f before)
  refine ⟨?_,?_⟩
  · intro h hh t ht j hj
    have hm : (⟨.output,h,0,t,j⟩:Address)∈Separate.forwardFootprint c f.flag :=
      List.mem_append_right _ (Source.output_cover c h t j hh ht hj)
    exact (finish_real_read c f.tensor _ admit fresh _ before _ hm).trans (source.output h hh t ht j hj)
  · intro final hf h hh e he j hj
    change f.finalState=some final at hf
    have flag : f.flag=Bool.true := by rw [←f.finalFlag,hf];rfl
    have hm : (⟨.finalState,h,0,e,j⟩:Address)∈Separate.forwardFootprint c f.flag := by
      rw [flag];exact List.mem_append_left _ (Source.final_cover c h e j hh he hj)
    have hw:=finish_real_read c f.tensor _ admit fresh
      (Source.afterO realOps c scale tree f.flag (forwardRealHeap c f before)) before _ hm
    have hv:=hw.trans (source.finalState flag h hh e he j hj)
    simpa only [realForwardReturn,cellAddress,ForwardObjects.tensor,hf,Option.getD_some,Address.offset] using hv

theorem backward_real_values (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (complete : tree.Complete c.keyBlocks) (priorDevice : Nat) (before later : RealMemory)
    (checks : backwardChecks c f b=Bool.true) (fresh : SeparateStores b.tensor (Separate.backwardFootprint c))
    (saved : Separate.Saved realOps c scale tree f.flag (forwardRealHeap c f before) (backwardRealHeap c f b later)) :
    RealBackwardValues c (Separate.joined (forwardRealHeap c f before) (backwardRealHeap c f b later)) scale
      (realBackwardReturn c f b scale tree priorDevice later) := by
  have admit:=backward_store_admission c f b checks
  have source:=Separate.backward_real c scale tree complete f.flag
    (forwardRealHeap c f before) (backwardRealHeap c f b later) saved
  refine ⟨?_,?_,?_⟩
  · intro h hh t ht e he
    have hm : (⟨.dq,h,0,t,e⟩:Address)∈Separate.backwardFootprint c := by
      simp only [Separate.backwardFootprint,List.mem_append]
      exact Or.inl (Or.inr (Source.dq_cover c h t e hh ht he))
    exact (finish_real_read c b.tensor _ admit fresh _ (prepareReal b later) _ hm).trans (source.query h hh t ht e he)
  · intro h hh t ht e he
    have hm : (⟨.dk,h,0,t,e⟩:Address)∈Separate.backwardFootprint c := by
      simp only [Separate.backwardFootprint,List.mem_append]
      exact Or.inl (Or.inr (Source.dk_cover c h t e hh ht he))
    exact (finish_real_read c b.tensor _ admit fresh _ (prepareReal b later) _ hm).trans (source.key h hh t ht e he)
  · intro h hh t ht j hj
    have hm : (⟨.dv,h,0,t,j⟩:Address)∈Separate.backwardFootprint c :=
      List.mem_append_right _ (Source.sum_cover c h t j hh ht hj)
    exact (finish_real_read c b.tensor _ admit fresh _ (prepareReal b later) _ hm).trans (source.value h hh t ht j hj)

/-- Pair the actual returned Tensor values with an arbitrary primal variation. -/
def returnedRealPairing (c : Shape) (b : RealBackwardReturn) (d : MathTarget.Variation) : ℝ :=
  (∑ h∈Finset.range c.heads,∑ t∈Finset.range c.T,∑ e∈Finset.range c.K,
    b.memory (b.dq.dataPointer+4*c.keyOffset h t e)*d.q h t e)+
  (∑ h∈Finset.range c.heads,∑ t∈Finset.range c.T,∑ e∈Finset.range c.K,
    b.memory (b.dk.dataPointer+4*c.keyOffset h t e)*d.k h t e)+
  (∑ h∈Finset.range c.heads,∑ t∈Finset.range c.T,∑ j∈Finset.range c.V,
    b.memory (b.dv.dataPointer+4*c.valueOffset h t j)*d.v h t j)

theorem returned_real_pairing {c : Shape} {xs : Inputs ℝ} {scale : ℝ} {b : RealBackwardReturn}
    (p : RealBackwardValues c xs scale b) (d : MathTarget.Variation) :
    returnedRealPairing c b d=MathTarget.gradientPairing c xs (detachedVariation d) scale Bool.true := by
  unfold returnedRealPairing MathTarget.gradientPairing
  simp only [detachedVariation,↓reduceIte,mul_zero,Finset.sum_const_zero,add_zero]
  apply congrArg₂ (·+·)
  · apply congrArg₂ (·+·)
    · apply Finset.sum_congr rfl
      intro h hh
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro e he
      rw [p.query h (Finset.mem_range.mp hh) t (Finset.mem_range.mp ht) e (Finset.mem_range.mp he)]
    · apply Finset.sum_congr rfl
      intro h hh
      apply Finset.sum_congr rfl
      intro t ht
      apply Finset.sum_congr rfl
      intro e he
      rw [p.key h (Finset.mem_range.mp hh) t (Finset.mem_range.mp ht) e (Finset.mem_range.mp he)]
  · apply Finset.sum_congr rfl
    intro h hh
    apply Finset.sum_congr rfl
    intro t ht
    apply Finset.sum_congr rfl
    intro j hj
    rw [p.value h (Finset.mem_range.mp hh) t (Finset.mem_range.mp ht) j (Finset.mem_range.mp hj)]

def RealVJP (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (b : RealBackwardReturn) : Prop :=
  (∀ d : MathTarget.Variation,MathTarget.fullVariation c xs (detachedVariation d) scale Bool.true=returnedRealPairing c b d) ∧
  ∀ (d : MathTarget.Variation) (z : ℝ),
    MathTarget.loss c (MathTarget.perturb xs (detachedVariation d) z) scale Bool.true=
      MathTarget.loss c xs scale Bool.true+z*returnedRealPairing c b d+
      z^2*MathTarget.secondCoefficient c xs (detachedVariation d) scale Bool.true+
      z^3*MathTarget.thirdCoefficient c xs (detachedVariation d) scale

theorem real_vjp {c : Shape} {xs : Inputs ℝ} {scale : ℝ} {b : RealBackwardReturn}
    (p : RealBackwardValues c xs scale b) : RealVJP c xs scale b := by
  constructor
  · intro d
    exact (detached_vjp c xs d scale).trans (returned_real_pairing p d).symm
  · intro d z
    rw [returned_real_pairing p]
    exact MathTarget.loss_gradient_polynomial c xs (detachedVariation d) scale Bool.true z

theorem backward_real_preserves (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (later : RealMemory) (checks : backwardChecks c f b=Bool.true)
    (t : Tensor) (fresh : ∀ a∈Separate.backwardFootprint c,Disjoint (b.tensor a.region) t)
    (zeroFresh : b.incomingPresent=Bool.false → Disjoint b.incoming t)
    (point : Nat) (hlo : t.storageBase≤point) (hhi : point<t.storageBase+t.storageBytes) :
    backwardRealMemory c f b scale tree later point=later point := by
  unfold backwardRealMemory
  rw [finish_real_preserves c b.tensor _ (backward_store_admission c f b checks) _ _ t fresh point hlo hhi]
  cases hi:b.incomingPresent
  · simp only [prepareReal,hi,↓reduceIte]
    exact zero_real_preserves _ b.incoming t (backward_pointer c f b checks .incoming (by simp))
      (zeroFresh hi) later point hlo hhi
  · simp only [prepareReal,hi,↓reduceIte]

#axiomsClean forward_real_values
#axiomsClean backward_real_values
#axiomsClean returned_real_pairing
#axiomsClean real_vjp
#axiomsClean backward_real_preserves
#specNonCircular RealForwardValues avoiding [forwardRealCall,forwardRealMemory,forward_real_values]
#specNonCircular RealBackwardValues avoiding [backwardRealCall,backwardRealMemory,backward_real_values]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
