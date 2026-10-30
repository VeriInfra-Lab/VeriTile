import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Single
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Real
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section
open Classical
set_option maxHeartbeats 6000000

def eval (data : RegionName → Nat → ℝ) : Word → ℝ
  | .zero => 0 | .leaf r o => data r o
  | .cast _ _ x => eval data x
  | .mul _ a b => eval data a * eval data b
  | .add _ a b => eval data a + eval data b

/-- Mathematical LoRA target, independent of the source, tree, and scheduler. -/
def target (data : RegionName → Nat → ℝ) (g : Geometry) (raw : Nat → Int) (b n : Nat) : ℝ :=
  if logicalIndex g raw b=-1 then data "Y" (yoff g b n) else
    (∑ k : Fin g.K, data "X" (xoff g b k.val) *
      data "W" (woff g (logicalIndex g raw b).toNat n k.val)) +
    (if g.add then data "Y" (yoff g b n) else 0)

theorem cast_eval (cfg : FP.Config) (src dst : OutputFormat) (x : Word) (data : RegionName → Nat → ℝ) :
    eval data (castWord cfg src dst x)=eval data x := by
  unfold castWord
  split
  · rfl
  · cases x <;> rfl

theorem tree_eval (cfg : FP.Config) (fmt : OutputFormat) (xs : Fin n → Word)
    (tree : ReductionTree n) (data : RegionName → Nat → ℝ) :
    eval data (treeValue cfg fmt xs tree)=
      (tree.leaves.map (fun i => i.elim 0 (fun k => eval data (xs k)))).sum := by
  induction tree with
  | input i => simp [treeValue,ReductionTree.leaves,eval]
  | zero => simp [treeValue,ReductionTree.leaves,eval]
  | add a b ha hb => simp [treeValue,addWord,eval,ReductionTree.leaves,ha,hb,List.sum_append]

theorem plan_eval (cfg : FP.Config) (fmt : OutputFormat) (xs : Fin n → Word)
    (plan : ReductionPlan n) (data : RegionName → Nat → ℝ) :
    eval data (treeValue cfg fmt xs plan.tree)=∑ k : Fin n,eval data (xs k) := by
  rw [tree_eval]
  have h := List.Perm.sum_eq (plan.valid.map (fun i => i.elim (0 : ℝ) (fun k => eval data (xs k))))
  rw [h]
  simp [List.map_append,List.map_map,List.sum_append,Function.comp_def,←Fin.sum_univ_def]

theorem value_eq_target (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int)
    (v : Valid g raw) (data : RegionName → Nat → ℝ) (b n : Nat) :
    eval data ((resultCell plans g raw b n).read .real)=target data g raw b n := by
  unfold resultCell target
  split
  · simp [entry,Cell.read_same,eval]
  · simp only [Cell.read_same,value,cast_eval,plan_eval]
    have hs : (∑ k : Fin g.block,eval data (product g raw b n k))=
        ∑ k : Fin g.K,data "X" (xoff g b k.val)*data "W" (woff g (logicalIndex g raw b).toNat n k.val) := by
      simp only [product,eval,cast_eval]
      have hp : ∀ k : Fin g.block,
          (eval data (if k.val<g.K then .leaf "X" (xoff g b k.val) else .zero) *
          eval data (if k.val<g.K then .leaf "W" (woff g (logicalIndex g raw b).toNat n k.val) else .zero))=
          if k.val<g.K then data "X" (xoff g b k.val)*data "W" (woff g (logicalIndex g raw b).toNat n k.val) else 0 := by
        intro k; split <;> simp_all [eval]
      simp_rw [hp]
      rw [TritonBenchG.LoraExpandGemv.Full.sum_fin_ite_range g.block g.K v.tile.1
        (fun k => data "X" (xoff g b k)*data "W" (woff g (logicalIndex g raw b).toNat n k))]
      exact (Fin.sum_univ_eq_sum_range
        (fun k => data "X" (xoff g b k)*data "W" (woff g (logicalIndex g raw b).toNat n k)) g.K).symm
    cases ha : g.add <;> simp [ha,eval,cast_eval,plan_eval,hs]

#axiomsClean cast_eval
#axiomsClean tree_eval
#axiomsClean plan_eval
#axiomsClean value_eq_target
#specNonCircular target avoiding [Branches.source,FP.Structural.exec,value]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Real
