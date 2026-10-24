import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.PhaseAPI
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.LossPolynomial
namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section
open HostABI PhaseAPI
open scoped BigOperators
namespace Calls

def ForwardMetadata (args : CallMeta c) (alloc : ForwardBuffers caller) (m : Memory α) : Prop :=
  let f:=PhaseAPI.forwardReturn args alloc m
  f.output.dimensions=shapeValue c ∧ f.output.fp32=Bool.true ∧ f.output.device=args.currentDevice ∧
  f.finalState.isSome=args.finalRequested ∧ f.savedQ=args.q ∧ f.savedK=args.k ∧ f.savedV=args.v ∧
  f.savedHistory.dimensions=shapeHistory c ∧ f.output.storage∉caller

def BackwardMetadata (args : CallMeta c) (alloc : BackwardBuffers caller) (m : Memory α) : Prop :=
  let b:=PhaseAPI.backwardReturn args alloc m
  b.dq.dimensions=shapeKey c ∧ b.dk.dimensions=shapeKey c ∧ b.dv.dimensions=shapeValue c ∧
  b.dq.fp32=Bool.true ∧ b.dk.fp32=Bool.true ∧ b.dv.fp32=Bool.true ∧
  b.dq.device=args.currentDevice ∧ b.dk.device=args.currentDevice ∧ b.dv.device=args.currentDevice ∧
  b.initialGradient=none ∧ b.flagGradient=none ∧
  b.dq.storage∉caller ∧ b.dk.storage∉caller ∧ b.dv.storage∉caller

theorem forward_metadata (args : CallMeta c) (alloc : ForwardBuffers caller) (m : Memory α) :
    ForwardMetadata args alloc m := by
  refine ⟨rfl,rfl,rfl,?_,rfl,rfl,rfl,rfl,alloc.fresh 1⟩
  cases h:args.finalRequested <;> simp [PhaseAPI.forwardReturn,h]

theorem backward_metadata (args : CallMeta c) (alloc : BackwardBuffers caller) (m : Memory α) :
    BackwardMetadata args alloc m :=
  ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,alloc.fresh 1,alloc.fresh 2,alloc.fresh 4⟩

/-- Forward has only its own entry state, allocations and normalized inputs.
The unused forward bases are arbitrary ghost slots, with no connection to any
later cotangent or backward allocation. The backward call is universally
quantified afterwards, uses its own binding and its own entry-memory frame. -/
def Post (impl spec : Ops α) (c : Shape) (implScale specScale : α) (tree : SumTree)
    (args : CallMeta c) (alloc : ForwardBuffers caller) (fb : Binding c) (before : Memory α) : Prop :=
  PhaseAPI.ForwardPost spec c (inputsOfHeap (fb.view impl.zero before)) specScale args.finalRequested
    (PhaseAPI.forwardReturn args alloc (Separate.forwardPhysical impl c fb implScale tree args.finalRequested before)) ∧
  ForwardMetadata args alloc (Separate.forwardPhysical impl c fb implScale tree args.finalRequested before) ∧
  Separate.forwardPhysical impl c fb implScale tree args.finalRequested before=
    Separate.forwardPhysical spec c fb specScale tree args.finalRequested before ∧
  (∀ point,(∀ a∈Separate.forwardFootprint c args.finalRequested,point≠fb.address a) →
    Separate.forwardPhysical impl c fb implScale tree args.finalRequested before point=before point) ∧
  ∀ (laterCaller : List Nat) (ba : BackwardBuffers laterCaller) (g : Cotangents args)
      (b : Binding c) (later : Memory α), BackwardBinding args alloc ba g b →
    Separate.Saved impl c implScale tree args.finalRequested (fb.view impl.zero before) (b.view impl.zero later) →
    PhaseAPI.BackwardPost spec c (Separate.joined (fb.view impl.zero before) (b.view impl.zero later)) specScale tree
      (PhaseAPI.backwardReturn args ba (Separate.backwardPhysical impl c b implScale tree later)) ∧
    BackwardMetadata args ba (Separate.backwardPhysical impl c b implScale tree later) ∧
    Separate.backwardPhysical impl c b implScale tree later=Separate.backwardPhysical spec c b specScale tree later ∧
    (∀ point,(∀ a∈Separate.backwardFootprint c,point≠b.address a) →
      Separate.backwardPhysical impl c b implScale tree later point=later point)

theorem post (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree) (complete : tree.Complete c.keyBlocks)
    (args : CallMeta c) (alloc : ForwardBuffers caller) (fb : Binding c)
    (bound : ForwardBinding args alloc fb) (before : Memory α) :
    Post ops ops c scale scale tree args alloc fb before := by
  refine ⟨forward_handles ops c scale tree args alloc fb bound before,forward_metadata _ _ _,rfl,
    Separate.forward_physical_frame ops c fb scale tree args.finalRequested before,?_⟩
  intro laterCaller ba g b later bind saved
  exact ⟨backward_handles ops c scale tree complete args alloc ba g fb b bind before later saved,
    backward_metadata _ _ _,rfl,Separate.backward_physical_frame ops c b scale tree later⟩

structure RealForwardPost (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (returned : ForwardReturn ℝ) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    tensorRead returned.memory returned.output (c.valueOffset h t j)=MathTarget.output c xs scale Bool.true h t j
  finalState : ∀ f,returned.finalState=some f → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    tensorRead returned.memory f (c.stateOffset h e j)=MathTarget.finalState c xs Bool.true h e j

structure RealBackwardPost (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (returned : BackwardReturn ℝ) : Prop where
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    tensorRead returned.memory returned.dq (c.keyOffset h t e)=MathTarget.queryGradient c xs scale Bool.true h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    tensorRead returned.memory returned.dk (c.keyOffset h t e)=MathTarget.keyGradient c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    tensorRead returned.memory returned.dv (c.valueOffset h t j)=MathTarget.valueGradient c xs scale h t j

theorem forward_real {c : Shape} {xs : Inputs ℝ} {scale : ℝ} {flag : Bool} {f : ForwardReturn ℝ}
    (p : PhaseAPI.ForwardPost realOps c xs scale flag f) : RealForwardPost c xs scale f := by
  refine ⟨?_,?_⟩
  · intro h hh t ht j hj
    exact (p.output h hh t ht j hj).trans (RealScalar.reference_output c xs scale h t j ht hj)
  · intro f hf h hh e he j hj
    exact (p.finalState f hf h hh e he j hj).trans (RealScalar.reference_final c xs h e j he hj)

theorem backward_real {c : Shape} {xs : Inputs ℝ} {scale : ℝ} {tree : SumTree} {b : BackwardReturn ℝ}
    (complete : tree.Complete c.keyBlocks) (p : PhaseAPI.BackwardPost realOps c xs scale tree b) :
    RealBackwardPost c xs scale b := by
  refine ⟨?_,?_,?_⟩
  · intro h hh t ht e he
    exact (p.query h hh t ht e he).trans (RealScalar.reference_query c xs scale h t e ht he)
  · intro h hh t ht e he
    exact (p.key h hh t ht e he).trans (RealScalar.reference_key c xs scale h t e ht he)
  · intro h hh t ht j hj
    exact (p.value h hh t ht j hj).trans (RealScalar.reference_value c xs scale tree complete h t j ht hj)

/-- Pair actual returned gradient tensors with a primal variation. Seed is
detached in the source wrapper, so no seed-gradient handle participates. -/
def returnedPairing (c : Shape) (b : BackwardReturn ℝ) (d : MathTarget.Variation) : ℝ :=
  (∑ h∈Finset.range c.heads,∑ t∈Finset.range c.T,∑ e∈Finset.range c.K,
    tensorRead b.memory b.dq (c.keyOffset h t e)*d.q h t e)+
  (∑ h∈Finset.range c.heads,∑ t∈Finset.range c.T,∑ e∈Finset.range c.K,
    tensorRead b.memory b.dk (c.keyOffset h t e)*d.k h t e)+
  (∑ h∈Finset.range c.heads,∑ t∈Finset.range c.T,∑ j∈Finset.range c.V,
    tensorRead b.memory b.dv (c.valueOffset h t j)*d.v h t j)

theorem returned_pairing {c : Shape} {xs : Inputs ℝ} {scale : ℝ} {b : BackwardReturn ℝ}
    (p : RealBackwardPost c xs scale b) (d : MathTarget.Variation) :
    returnedPairing c b d=MathTarget.gradientPairing c xs (detachedVariation d) scale Bool.true := by
  unfold returnedPairing MathTarget.gradientPairing
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

def LossPost (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (b : BackwardReturn ℝ) : Prop :=
  ∀ (d : MathTarget.Variation) (z : ℝ),
    MathTarget.loss c (MathTarget.perturb xs (detachedVariation d) z) scale Bool.true=
      MathTarget.loss c xs scale Bool.true+z*returnedPairing c b d+
      z^2*MathTarget.secondCoefficient c xs (detachedVariation d) scale Bool.true+
      z^3*MathTarget.thirdCoefficient c xs (detachedVariation d) scale

theorem loss_post {c : Shape} {xs : Inputs ℝ} {scale : ℝ} {b : BackwardReturn ℝ}
    (p : RealBackwardPost c xs scale b) : LossPost c xs scale b := by
  intro d z
  rw [returned_pairing p]
  exact MathTarget.loss_gradient_polynomial c xs (detachedVariation d) scale Bool.true z

def RealPost (c : Shape) (scale : ℝ) (tree : SumTree)
    (args : CallMeta c) (alloc : ForwardBuffers caller) (fb : Binding c) (before : Memory ℝ) : Prop :=
  RealForwardPost c (inputsOfHeap (fb.view 0 before)) scale
    (PhaseAPI.forwardReturn args alloc (Separate.forwardPhysical realOps c fb scale tree args.finalRequested before)) ∧
  ForwardMetadata args alloc (Separate.forwardPhysical realOps c fb scale tree args.finalRequested before) ∧
  (∀ point,(∀ a∈Separate.forwardFootprint c args.finalRequested,point≠fb.address a) →
    Separate.forwardPhysical realOps c fb scale tree args.finalRequested before point=before point) ∧
  ∀ (laterCaller : List Nat) (ba : BackwardBuffers laterCaller) (g : Cotangents args)
      (b : Binding c) (later : Memory ℝ), BackwardBinding args alloc ba g b →
    Separate.Saved realOps c scale tree args.finalRequested (fb.view 0 before) (b.view 0 later) →
    let xs:=Separate.joined (fb.view 0 before) (b.view 0 later)
    let returned:=PhaseAPI.backwardReturn args ba (Separate.backwardPhysical realOps c b scale tree later)
    RealBackwardPost c xs scale returned ∧ LossPost c xs scale returned ∧
    BackwardMetadata args ba (Separate.backwardPhysical realOps c b scale tree later) ∧
    (∀ point,(∀ a∈Separate.backwardFootprint c,point≠b.address a) →
      Separate.backwardPhysical realOps c b scale tree later point=later point)

theorem real_post (c : Shape) (scale : ℝ) (tree : SumTree) (complete : tree.Complete c.keyBlocks)
    (args : CallMeta c) (alloc : ForwardBuffers caller) (fb : Binding c)
    (bound : ForwardBinding args alloc fb) (before : Memory ℝ) : RealPost c scale tree args alloc fb before := by
  have p:=post realOps c scale tree complete args alloc fb bound before
  refine ⟨forward_real p.1,p.2.1,p.2.2.2.1,?_⟩
  intro laterCaller ba g b later bind saved
  have bpost:=p.2.2.2.2 laterCaller ba g b later bind saved
  have real:=backward_real complete bpost.1
  exact ⟨real,loss_post real,bpost.2.1,bpost.2.2.2⟩

#axiomsClean post
#axiomsClean returned_pairing
#axiomsClean loss_post
#axiomsClean real_post
end Calls
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
