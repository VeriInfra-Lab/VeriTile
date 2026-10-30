import bench.repairs.tritonbench_g.fused_activation.Proofs.Rewrite
import bench.repairs.tritonbench_g.fused_activation.Proofs.Graph
import VeriTile.Triton.Float.ScalarOpsLaws

/-! Native equivalence keeps the selected multiply/add contraction as FMA.
Only the final bias addition is commuted. Primitive interpretation, including
FMA and sigmoid expansion, is shared by both sides and is not an experiment. -/
namespace VeriTile.Bench.Repairs.FusedActivation.NativeRewrite
open VeriTile.Triton FP.Structural FP.Guarded
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

def observeValue (ops : Graph.Operations) : (d : TileDType) → Value Graph.Expr d → Value Graph.Word d
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => Graph.eval ops
  | .nat | .int | .bool | .ptr | .blockPtr => id

def observeCell (ops : Graph.Operations) : Cell Graph.Expr → Cell Graph.Word
  | .mk d v => .mk d (observeValue ops d v)

def observeMemory (ops : Graph.Operations) (mem : RegionName → Nat → Cell Graph.Expr) :=
  fun r o => observeCell ops (mem r o)

/-- Bind only the native add port to the selected concrete FP32 model.
FMA, sigmoid, and every other native port retain their existing interpretation. -/
def AddBinding (ops : Graph.Operations) (cfg : FP.Config) : Prop :=
  ∀ a b, ops.add32 a b =
    (FP.add cfg (⟨a⟩ : FP.Value .fp32) (⟨b⟩ : FP.Value .fp32)).bits

/-- The shared strict FP32 theorem applies to every pair of native words.
No claim about independently rounded multiply/add appears here. -/
theorem add_comm (ops : Graph.Operations) (cfg : FP.Config)
    (binding : AddBinding ops cfg) (fused bias : Graph.Word) :
    ops.add32 fused bias = ops.add32 bias fused := by
  rw [binding fused bias, binding bias fused]
  exact congrArg FP.Value.bits (FP.ScalarOpsLaws.add_comm cfg _ _)

private theorem observed_scatter (ops : Graph.Operations) {ι : Type}
    (indices : List ι) (active : ι → Prop) [DecidablePred active]
    (output : RegionName) (address : ι → Nat) (lhs rhs : ι → Cell Graph.Expr)
    (values : ∀ j, active j → observeCell ops (lhs j) = observeCell ops (rhs j))
    (before before' : RegionName → Nat → Cell Graph.Expr)
    (same : observeMemory ops before = observeMemory ops before') :
    observeMemory ops (indices.foldl (fun mem j => if active j then fun r o =>
      if r=output ∧ o=address j then lhs j else mem r o else mem) before) =
    observeMemory ops (indices.foldl (fun mem j => if active j then fun r o =>
      if r=output ∧ o=address j then rhs j else mem r o else mem) before') := by
  induction indices generalizing before before' with
  | nil => exact same
  | cons j rest ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hj : active j
    · funext r o
      simp only [hj, ↓reduceIte, observeMemory]
      by_cases target : r=output ∧ o=address j
      · simp only [target]
        exact values j hj
      · simpa [target, observeMemory] using congrFun (congrFun same r) o
    · simpa [hj] using same

/-- Source success and all typed native memory observations. Raw load leaves
bind the selected lowering to actual words, including arbitrary special words;
only active outer-add operands carry the transformation's finite domain. -/
theorem memory_equiv (rest : Algebra Graph.Expr)
    (ops : Graph.Operations) (cfg : FP.Config) (binding : AddBinding ops cfg)
    (D : Domain Graph.Word)
    (p : Parameters) (W : Nat) (a : Bool) (s : State Graph.Expr)
    (xs biases inputs : Nat → Graph.Word)
    (hx : ∀ j : TileIndex [p.block], active p s j →
      (s.mem p.output (offset p s j)).read .real = .leaf (xs (offset p s j)))
    (hb : ∀ j : TileIndex [p.block], active p s j →
      (s.mem p.y (offset p s j % W)).read .real = .leaf (biases (offset p s j % W)))
    (hi : ∀ j : TileIndex [p.block], active p s j →
      (s.mem p.x (offset p s j)).read .real = .leaf (inputs (offset p s j)))
    (_domain : ∀ j : TileIndex [p.block], active p s j →
      D .finite (ops.fmaRN32 (inputs (offset p s j)) Graph.half (xs (offset p s j))) ∧
      D .finite (biases (offset p s j % W))) :
    observeMemory ops (execution (Graph.algebra rest) p W (1/2) a s).val.mem =
      observeMemory ops (Rewrite.commutedExecution (Graph.algebra rest) p W (1/2) a s).val.mem := by
  rw [execution_memory, Rewrite.commuted_memory]
  unfold expectedMemory Rewrite.commutedMemory
  apply observed_scatter ops _ (active p s) p.output (offset p s) _ _ ?_ s.mem s.mem rfl
  intro j hj
  have commute := add_comm ops cfg binding
    (ops.fmaRN32 (inputs (offset p s j)) Graph.half (xs (offset p s j)))
    (biases (offset p s j % W))
  simp only [resultCell, Rewrite.commutedCell, Graph.algebra, id_eq,
    hx j hj, hb j hj, hi j hj, observeCell, observeValue]
  cases a <;> simp [scalar, Rewrite.commutedScalar, Graph.binary, Graph.unary,
    Graph.eval, commute]

#axiomsClean add_comm
#axiomsClean memory_equiv
end
end VeriTile.Bench.Repairs.FusedActivation.NativeRewrite
