import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeInputs
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeAccess
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeTagsProducer
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeTagsConsumer

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeImplements
open VeriTile Triton FP.Structural NativeInstructions
open NativePorts (Word Expr)
open NativeIO (producer consumer attention output)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem observe_real {α β : Type} [Inhabited α] (f : α → β) (cell : Cell α)
    (tag : NativeTags.cellDtype cell=.real) :
    mapCell f cell=.mk .real (f (FPStorage.payload cell)) := by
  cases cell with
  | mk dtype value =>
    change dtype=.real at tag
    subst dtype
    rfl

theorem observe_fp32 {α β : Type} [Inhabited α] (f : α → β) (cell : Cell α)
    (tag : NativeTags.cellDtype cell=.fp32) :
    mapCell f cell=.mk .fp32 (f (FPStorage.payload cell)) := by
  cases cell with
  | mk dtype value =>
    change dtype=.fp32 at tag
    subst dtype
    rfl

theorem producer_outside (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (producer c I w scale).Inputs) (r : RegionName) (off : Nat)
    (outside : (producer c I w scale).Outside values r off) : NativeWhole.Outside c r off := by
  constructor
  · by_cases same : r="attention"
    · right
      by_contra within
      have hi : off<Host.attentionSize c := by omega
      subst r
      cases tag : NativeTags.producerExplicit c off with
      | «false» =>
        have excluded := outside.1 (0 : Fin 2) (⟨off,hi⟩,⟨⟩) tag
        exact excluded.elim (fun h=>h rfl) (fun h=>h rfl)
      | «true» =>
        have excluded := outside.1 (1 : Fin 2) (⟨off,hi⟩,⟨⟩) tag
        exact excluded.elim (fun h=>h rfl) (fun h=>h rfl)
    · exact Or.inl same
  · by_cases empty : c.heads*c.T=0
    · exact Or.inl empty
    · right
      by_cases small : c.K≤256
      · exact Or.inl small
      · right
        by_cases same : r="scratch"
        · right
          by_contra within
          have hi : off<NativeIO.scratchSize c := by
            simp only [NativeIO.scratchSize,if_neg (not_or_intro empty small)]
            omega
          subst r
          have excluded := outside.2 (0 : Fin 1) (⟨off,hi⟩,⟨⟩) True.intro
          exact excluded.elim (fun h=>h rfl) (fun h=>h rfl)
        · exact Or.inl same

theorem consumer_outside (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (consumer c I w scale).Inputs) (r : RegionName) (off : Nat)
    (outside : (consumer c I w scale).Outside values r off) : r≠"o" ∨ Host.outputSize c≤off := by
  by_cases same : r="o"
  · right
    by_contra within
    have hi : off<Host.outputSize c := by omega
    subst r
    have excluded := outside.1 (0 : Fin 1) (⟨off,hi⟩,⟨⟩) True.intro
    exact excluded.elim (fun h=>h rfl) (fun h=>h rfl)
  · exact Or.inl same

theorem producer_correct (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    (producer c I w scale).Implements (attention c I w scale) := by
  intro inputs _ initial bound
  constructor
  · intro bounds within
    exact NativeAccess.producer (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag bounds
      (NativeBounds.producer_bounds c I w scale inputs bounds within) initial
  · obtain ⟨final,ran,values,frame⟩ := NativeWhole.producer (KernelProfile.algebra w scale) NativeStaging.dot
      c NativeRuntime.scaleTag initial (NativePorts.producerMemory c inputs)
      (NativeInputs.producer_read c I w scale inputs initial bound)
    refine ⟨final,ran,?_,?_⟩
    · intro channel i active
      fin_cases channel
      · change NativeTags.producerExplicit c i.1.val=Bool.false at active
        have tag := NativeTags.producer_dtype (KernelProfile.algebra w scale) NativeStaging.dot
          c NativeRuntime.scaleTag initial final ran i.1.val i.1.isLt
        simp only [active,Bool.false_eq_true,↓reduceIte] at tag
        change mapCell (fun expression=>(NativeStaging.lower expression).eval I) (final.mem "attention" i.1.val)=
          .mk .real (NativePorts.attentionWord c I w scale inputs i.1.val)
        rw [observe_real _ _ tag]
        exact congrArg (fun expression=>Cell.mk .real ((NativeStaging.lower expression).eval I))
          (values i.1.val i.1.isLt)
      · change NativeTags.producerExplicit c i.1.val=Bool.true at active
        have tag := NativeTags.producer_dtype (KernelProfile.algebra w scale) NativeStaging.dot
          c NativeRuntime.scaleTag initial final ran i.1.val i.1.isLt
        simp only [active,↓reduceIte] at tag
        change mapCell (fun expression=>(NativeStaging.lower expression).eval I) (final.mem "attention" i.1.val)=
          .mk .fp32 (NativePorts.attentionWord c I w scale inputs i.1.val)
        rw [observe_fp32 _ _ tag]
        exact congrArg (fun expression=>Cell.mk .fp32 ((NativeStaging.lower expression).eval I))
          (values i.1.val i.1.isLt)
    · intro r off outside
      exact frame r off (producer_outside c I w scale inputs r off outside)

theorem consumer_correct (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word) :
    (consumer c I w scale).Implements (output c I w scale) := by
  intro inputs _ initial bound
  constructor
  · intro bounds within
    exact NativeAccess.consumer (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag bounds
      (NativeBounds.consumer_bounds c I w scale inputs bounds within) initial
  · obtain ⟨final,ran,values,frame⟩ := NativeWhole.consumer (KernelProfile.algebra w scale) NativeStaging.dot
      c NativeRuntime.scaleTag initial
    refine ⟨final,ran,?_,?_⟩
    · intro channel i _
      fin_cases channel
      have tag := NativeTags.consumer_dtype (KernelProfile.algebra w scale) NativeStaging.dot
        c NativeRuntime.scaleTag initial final ran i.1.val i.1.isLt
      have observed := values "o" i.1.val
      change FPStorage.payload (final.mem "o" i.1.val)=
        Host.consumerReference (KernelProfile.algebra w scale) NativeStaging.dot c NativeRuntime.scaleTag
          (NativeScope.consumerMemory initial) "o" i.1.val at observed
      have returned := FPReturned.consumer_readback (KernelProfile.algebra w scale) NativeStaging.dot
        c NativeRuntime.scaleTag (NativeScope.consumerMemory initial) i.1.divNat.divNat i.1.divNat.modNat i.1.modNat
      rw [NativeInputs.flat_coordinate] at returned
      have expected := observed.trans (returned.trans
        (NativeInputs.consumer_value c I w scale inputs initial bound i.1.divNat.divNat i.1.divNat.modNat i.1.modNat))
      change mapCell (fun expression=>(NativeStaging.lower expression).eval I) (final.mem "o" i.1.val)=
        .mk .fp32 (NativePorts.outputWord c I w scale inputs i.1)
      rw [observe_fp32 _ _ tag]
      exact congrArg (fun expression=>Cell.mk .fp32 ((NativeStaging.lower expression).eval I)) expected
    · intro r off outside
      exact frame r off (consumer_outside c I w scale inputs r off outside)

#axiomsClean producer_correct
#axiomsClean consumer_correct
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeImplements
