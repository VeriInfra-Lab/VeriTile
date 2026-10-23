import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.RealScore
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Components

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def CarryEquivalent (left right : Carry α) : Prop :=
  left.maximum=right.maximum ∧ left.denominator=right.denominator ∧
    ∀ i, left.output i (owner i.2.1.val)=right.output i (owner i.2.1.val)

theorem math_add_chain_seed {k : Nat} (left right : Fin k → Replicas R) (seed : R) :
    mathArithmetic.add seed (chain mathArithmetic left right mathArithmetic.zero)=
      chain mathArithmetic left right seed := (math_chain_add left right seed).symm

theorem update_math (p : Parameters) (s : State (Replicas R)) (column : Int)
    (left right : Carry R) (same : CarryEquivalent left right) :
    CarryEquivalent (update mathArithmetic Bool.false p s column left)
      (update mathArithmetic Bool.true p s column right) := by
  rcases same with ⟨hm,hl,ho⟩
  have hs : Component.scores mathArithmetic Bool.false p s column=
      Component.scores mathArithmetic Bool.true p s column := by
    funext i o
    simp only [Component.scores,Bool.false_eq_true,ite_false,ite_true,scoreAccumulation_math]
    rw [math_fma,math_mul_comm]
  have hb : Component.maximum mathArithmetic Bool.false p s column=
      Component.maximum mathArithmetic Bool.true p s column := by
    funext i o
    simp only [Component.maximum,hs]
  have hp : Component.probability mathArithmetic Bool.false p s column=
      Component.probability mathArithmetic Bool.true p s column := by
    funext i o
    simp only [Component.probability,hs,hb]
  have hd : Component.denominator mathArithmetic Bool.false p s column=
      Component.denominator mathArithmetic Bool.true p s column := by
    funext i o
    simp only [Component.denominator,hp]
  have hn : Component.nextMaximum mathArithmetic Bool.false p s column left=
      Component.nextMaximum mathArithmetic Bool.true p s column right := by
    funext i o
    simp only [Component.nextMaximum,hm,hb]
  have ha : Component.alpha mathArithmetic Bool.false p s column left=
      Component.alpha mathArithmetic Bool.true p s column right := by
    funext i o
    simp only [Component.alpha,hm,hn]
  have he : Component.beta mathArithmetic Bool.false p s column left=
      Component.beta mathArithmetic Bool.true p s column right := by
    funext i o
    simp only [Component.beta,hb,hn]
  have hl' : Component.nextDenominator mathArithmetic Bool.false p s column left=
      Component.nextDenominator mathArithmetic Bool.true p s column right := by
    funext i o
    simp only [Component.nextDenominator,Bool.false_eq_true,ite_false,ite_true,hl,ha,he,hd]
    rw [math_fma]
    congr 1 <;> exact math_mul_comm _ _
  have hp' : Component.scaledProbability mathArithmetic Bool.false p s column left=
      Component.scaledProbability mathArithmetic Bool.true p s column right := by
    funext i o
    simp only [Component.scaledProbability,hp,he,hl']
  have hf : Component.scale mathArithmetic Bool.false p s column left=
      Component.scale mathArithmetic Bool.true p s column right := by
    funext i o
    simp only [Component.scale,Bool.false_eq_true,ite_false,ite_true,hl,ha,hl']
    exact math_mul_comm _ _
  rw [update_components,update_components]
  refine ⟨hn,hl',?_⟩
  intro i
  change Component.output mathArithmetic Bool.false p s column left i (owner i.2.1.val)=
    Component.output mathArithmetic Bool.true p s column right i (owner i.2.1.val)
  simp only [Component.output,Bool.false_eq_true,ite_false,ite_true,hp',tensor,binary,broadcast]
  rw [math_add_chain_seed]
  congr 1
  simp only [Component.scaledOutput,hf,ho]

theorem accumulation_math (p : Parameters) (s : State (Replicas R)) (visits : Nat) :
    CarryEquivalent (accumulation mathArithmetic Bool.false p s visits)
      (accumulation mathArithmetic Bool.true p s visits) := by
  induction visits with
  | zero => exact ⟨rfl,rfl,fun _ => rfl⟩
  | succ visits ih =>
    have succ (native : Bool) : accumulation mathArithmetic native p s (visits+1)=
        update mathArithmetic native p s (csrColumn p s (first p s+visits))
          (accumulation mathArithmetic native p s visits) := by
      simp [accumulation,List.range_succ,List.foldl_append]
    rw [succ,succ]
    exact update_math p s _ _ _ ih

theorem output_math (p : Parameters) (s : State (Replicas R)) (i : TileIndex [16,16]) :
    output mathArithmetic Bool.false p s i (owner i.2.1.val)=
      output mathArithmetic Bool.true p s i (owner i.2.1.val) :=
  (accumulation_math p s (visitCount p s)).2.2 i

#axiomsClean math_add_chain_seed
#axiomsClean update_math
#axiomsClean accumulation_math
#axiomsClean output_math
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
