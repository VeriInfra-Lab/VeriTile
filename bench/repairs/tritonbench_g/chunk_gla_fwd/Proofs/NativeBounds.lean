import bench.repairs.tritonbench_g.chunk_gla_fwd.NativeIO
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafety

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeBounds
open VeriTile Triton FP.Structural NativeInstructions
open NativePorts (Word Expr)
open NativeIO (producer consumer)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 1000000

theorem extent {n capacity : Nat} (h : ∀ i : Fin n,i.val<capacity) : n≤capacity := by
  by_contra beyond
  have inside := h ⟨capacity,by omega⟩
  change capacity<capacity at inside
  omega

structure Producer (c : Host.Dimensions) (bounds : RegionName → Nat) : Prop where
  query : c.heads*c.T*c.K≤bounds "q"
  key : c.heads*c.T*c.K≤bounds "k"
  gate : c.heads*c.T*c.K≤bounds "g"
  attention : Host.attentionSize c≤bounds "attention"
  scratch : NativeIO.scratchSize c≤bounds "scratch"

structure Consumer (c : Host.Dimensions) (bounds : RegionName → Nat) : Prop where
  query : c.heads*c.T*c.K≤bounds "q"
  gate : c.heads*c.T*c.K≤bounds "g"
  value : Host.outputSize c≤bounds "v"
  history : c.heads*c.NT*c.K*c.V≤bounds "h"
  attention : Host.attentionSize c≤bounds "attention"
  output : Host.outputSize c≤bounds "o"

theorem producer_bounds (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (producer c I w scale).Inputs) (bounds : RegionName → Nat)
    (hb : (producer c I w scale).InBounds bounds values) : Producer c bounds := by
  constructor
  · apply extent; intro i; exact hb.2.1 (0 : Fin 3) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.1 (1 : Fin 3) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.1 (2 : Fin 3) (i,⟨⟩) True.intro
  · apply extent
    intro i
    cases tag : NativeTags.producerExplicit c i.val with
    | «false» => exact hb.2.2.1 (0 : Fin 2) (i,⟨⟩) tag
    | «true» => exact hb.2.2.1 (1 : Fin 2) (i,⟨⟩) tag
  · apply extent; intro i; exact hb.2.2.2 (0 : Fin 1) (i,⟨⟩) True.intro

theorem consumer_bounds (c : Host.Dimensions) (I : ISA Word) (w : KernelProfile.Profile) (scale : Word)
    (values : (consumer c I w scale).Inputs) (bounds : RegionName → Nat)
    (hb : (consumer c I w scale).InBounds bounds values) : Consumer c bounds := by
  constructor
  · apply extent; intro i; exact hb.2.1 (0 : Fin 5) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.1 (1 : Fin 5) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.1 (2 : Fin 5) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.1 (3 : Fin 5) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.1 (4 : Fin 5) (i,⟨⟩) True.intro
  · apply extent; intro i; exact hb.2.2.1 (0 : Fin 1) (i,⟨⟩) True.intro

theorem inter (c : Host.Dimensions) (bounds : RegionName → Nat) (h : Producer c bounds) (scale : ℝ) :
    NativeLaunch.Bounds (.inter c.producer scale) bounds := by
  refine ⟨?_,h.attention⟩
  intro port member
  simp only [NativeLaunch.inputExtents,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl
  · exact h.query
  · exact h.key
  · exact h.gate

theorem intra (c : Host.Dimensions) (bounds : RegionName → Nat) (h : Producer c bounds) (scale : ℝ) :
    NativeLaunch.Bounds (.intra c.producer c.intraBK scale) bounds := inter c bounds h scale

theorem scratch (c : Host.Dimensions) (bounds : RegionName → Nat) (h : Producer c bounds)
    (nonempty : c.heads*c.T≠0) (large : ¬c.K≤256) : Host.scratchSize c≤bounds "scratch" := by
  simpa only [NativeIO.scratchSize,if_neg (not_or_intro nonempty large)] using h.scratch

theorem split (c : Host.Dimensions) (bounds : RegionName → Nat) (h : Producer c bounds) (scale : ℝ)
    (nonempty : c.heads*c.T≠0) (large : ¬c.K≤256) : NativeLaunch.Bounds (.split c.split scale) bounds :=
  ⟨(inter c bounds h scale).1,scratch c bounds h nonempty large⟩

theorem merge (c : Host.Dimensions) (bounds : RegionName → Nat) (h : Producer c bounds)
    (nonempty : c.heads*c.T≠0) (large : ¬c.K≤256) : NativeLaunch.Bounds (.merge c.split) bounds := by
  refine ⟨?_,h.attention⟩
  intro port member
  simp only [NativeLaunch.inputExtents,List.mem_singleton] at member
  subst port
  exact scratch c bounds h nonempty large

theorem output (c : Host.Dimensions) (bounds : RegionName → Nat) (h : Consumer c bounds) (scale : ℝ) :
    NativeLaunch.Bounds (.output c.consumer scale) bounds := by
  refine ⟨?_,h.output⟩
  intro port member
  simp only [NativeLaunch.inputExtents,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl
  · exact h.query
  · exact h.gate
  · exact h.value
  · simpa only [Host.Dimensions.consumer,Host.Dimensions.NT,Nat.mul_assoc] using h.history
  · exact h.attention

end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeBounds
