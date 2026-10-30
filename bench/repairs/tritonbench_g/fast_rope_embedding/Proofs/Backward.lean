import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Public

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs
noncomputable section

/-- Plain ctx attribute lookup, with current coefficient storage. No values,
snapshots, version checks, or forward-time coefficient equality are assumed. -/
structure BackwardPreparation {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (r : RopeEmbedding.Request) (before : Store) where
  stage : Stage api embed r before
  saved : Cache
  cosineLookup : stage.call.cosineInput=saved.cosine
  sineLookup : stage.call.sineInput=saved.sine

structure BackwardResult (Store : Type) where
  memory : Store
  gradients : RopeEmbedding.Native.Tensor × Option RopeEmbedding.Native.Tensor × Option RopeEmbedding.Native.Tensor
  outer : RopeEmbedding.Native.Tensor

def backwardCheck {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (r : RopeEmbedding.Request) (before : Store) (p : BackwardPreparation api embed r before) :=
  RopeEmbedding.Checked.check api embed r p.stage.state p.stage.call Bool.false Bool.false &&
    RopeEmbedding.Checked.cacheCheck r p.saved.groups p.saved.block p.saved.warps

def backwardResult {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : RopeEmbedding.Operations α) (r : RopeEmbedding.Request) (before : Store)
    (p : BackwardPreparation api embed r before) : BackwardResult Store :=
  ⟨(RopeEmbedding.Native.entry api embed project op r Bool.true p.stage.state p.stage.call).memory,
    (p.stage.inner.tensor,none,none),p.stage.call.output⟩

def backwardInvoke {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (project : α→β) (op : RopeEmbedding.Operations α) (r : RopeEmbedding.Request) (before : Store)
    (p : BackwardPreparation api embed r before) :=
  if backwardCheck api embed r before p then some (backwardResult api embed project op r before p) else none

def BackwardRealOutcome (api : RopeEmbedding.Memory.API ℝ Store Byte) (r : RopeEmbedding.Request) (before : Store)
    (p : BackwardPreparation api id r before) : Prop :=
  ∃ out,backwardInvoke api id id RopeEmbedding.ideal r before p=some out ∧
    out.gradients=(p.stage.inner.tensor,none,none) ∧
    RopeEmbedding.Checked.RealOutput api r Bool.true p.stage.state p.stage.call ⟨out.memory,out.outer⟩ ∧
    out.gradients.1.shape=[r.batches,r.sequence,r.heads,r.dimension] ∧
    (∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension →
      RopeEmbedding.Native.logicalRead api out.memory out.gradients.1 b t h c=
        RopeEmbedding.Checked.mathematicalTarget api r Bool.true before p.stage.call.input p.saved.cosine p.saved.sine b h t c) ∧
    (p.saved.groups,p.saved.block,p.saved.warps)=(r.groups,r.block,r.warps)

def BackwardFP32Outcome (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8)) (n : RopeEmbedding.Numerics.Primitives)
    (r : RopeEmbedding.Request) (before : Store) (p : BackwardPreparation api RopeEmbedding.Numerics.leaf r before) : Prop :=
  ∃ out,backwardInvoke api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) r before p=some out ∧
    out.gradients=(p.stage.inner.tensor,none,none) ∧
    RopeEmbedding.Checked.FP32Output api n r Bool.true p.stage.state p.stage.call ⟨out.memory,out.outer⟩ ∧
    out.gradients.1.shape=[r.batches,r.sequence,r.heads,r.dimension] ∧
    (∀ b h t c,b<r.batches → h<r.heads → t<r.sequence → c<r.dimension →
      RopeEmbedding.Native.logicalRead api out.memory out.gradients.1 b t h c=
        RopeEmbedding.Checked.machineTarget api n r Bool.true before p.stage.call.input p.saved.cosine p.saved.sine b h t c) ∧
    (p.saved.groups,p.saved.block,p.saved.warps)=(r.groups,r.block,r.warps)

theorem cache_objects {α : Type} [Inhabited α] (api : RopeEmbedding.Memory.API β Store Byte)
    (embed : β→α) (r : RopeEmbedding.Request) (before : Store) (p : BackwardPreparation api embed r before)
    (accepted : RopeEmbedding.Checked.check api embed r p.stage.state p.stage.call Bool.false Bool.false=Bool.true) :
    p.stage.call.cosine=p.saved.cosine ∧ p.stage.call.sine=p.saved.sine := by
  have unsqueezed := (RopeEmbedding.Checked.path_admitted api embed r p.stage.state p.stage.call Bool.false Bool.false accepted).2
  have views := p.stage.call.coefficientViews
  simp only [unsqueezed,Bool.false_eq_true,if_false] at views
  exact ⟨views.1.trans p.cosineLookup,views.2.trans p.sineLookup⟩

theorem whole_backward_real (api : RopeEmbedding.Memory.API ℝ Store Byte) (r : RopeEmbedding.Request) (before : Store)
    (p : BackwardPreparation api id r before) (accepted : backwardCheck api id r before p=Bool.true) :
    BackwardRealOutcome api r before p := by
  have admitted := accepted
  simp only [backwardCheck,Bool.and_eq_true] at admitted
  have whole := RopeEmbedding.Checked.whole_real api r Bool.true p.stage.state p.stage.call Bool.false Bool.false admitted.1
  have objects := cache_objects api id r before p admitted.1
  refine ⟨backwardResult api id id RopeEmbedding.ideal r before p,by simp [backwardInvoke,accepted],rfl,whole,p.stage.inner.shape,?_,
    RopeEmbedding.Checked.cache_admitted r p.saved.groups p.saved.block p.saved.warps admitted.2⟩
  intro b h t c hb hh ht hc
  change api.read _ (RopeEmbedding.Native.logicalAddress p.stage.inner.tensor b t h c)=_
  rw [p.stage.inner.address b h t c hb hh ht hc]
  have value := whole.1.logical b h t c hb hh ht hc
  rw [whole.1.returned,p.stage.entered,objects.1,objects.2] at value
  exact value

theorem whole_backward_fp32 (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8)) (n : RopeEmbedding.Numerics.Primitives)
    (r : RopeEmbedding.Request) (before : Store) (p : BackwardPreparation api RopeEmbedding.Numerics.leaf r before)
    (codec : ∀ w k,api.encode w k=RopeEmbedding.Numerics.wordByte w k)
    (accepted : backwardCheck api RopeEmbedding.Numerics.leaf r before p=Bool.true) : BackwardFP32Outcome api n r before p := by
  have admitted := accepted
  simp only [backwardCheck,Bool.and_eq_true] at admitted
  have whole := RopeEmbedding.Checked.whole_fp32 api n r Bool.true p.stage.state p.stage.call Bool.false Bool.false codec admitted.1
  have objects := cache_objects api RopeEmbedding.Numerics.leaf r before p admitted.1
  refine ⟨backwardResult api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word (RopeEmbedding.Numerics.operations n) r before p,
    by simp [backwardInvoke,accepted],rfl,whole,p.stage.inner.shape,?_,
    RopeEmbedding.Checked.cache_admitted r p.saved.groups p.saved.block p.saved.warps admitted.2⟩
  intro b h t c hb hh ht hc
  change api.read _ (RopeEmbedding.Native.logicalAddress p.stage.inner.tensor b t h c)=_
  rw [p.stage.inner.address b h t c hb hh ht hc]
  have value := whole.1.logical b h t c hb hh ht hc
  rw [whole.1.returned,p.stage.entered,objects.1,objects.2] at value
  exact value

#axiomsClean cache_objects
#axiomsClean whole_backward_real
#axiomsClean whole_backward_fp32
#specNonCircular BackwardPreparation avoiding [backwardInvoke,backwardResult,RopeEmbedding.Native.entry,RopeEmbedding.Memory.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRopeEmbedding
