import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Statistics
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeCoverage
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
namespace Samples
export VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
  (ideal advanceSamples sampleRecurrence recurrence_samples_perm collected_samples_range emitted_range)
end Samples
open Samples
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def contribution (phase : Bool) (mean x : ℝ) : ℝ := if phase then (x-mean)^2 else x

theorem advance_real (rest : Algebra ℝ) (phase : Bool) (s : State ℝ)
    (X : RegionName) (D B k : Nat) (mean : ℝ) (samples : Fin B → List ℝ) (i : Fin B) :
    advance (ideal rest) phase s X D B k mean (fun j => ((samples j).map (contribution phase mean)).sum) i =
      ((advanceSamples s X D B k samples i).map (contribution phase mean)).sum := by
  cases phase <;> simp only [advance,loaded,advanceSamples,contribution,ideal,zero]
  all_goals split_ifs <;> simp_all [List.sum_append,pow_two,contribution]

theorem recurrence_real (rest : Algebra ℝ) (phase : Bool) (s : State ℝ)
    (X : RegionName) (D B start count : Nat) (mean : ℝ) (samples : Fin B → List ℝ) :
    recurrence (ideal rest) phase s X D B mean start count
      (fun i => ((samples i).map (contribution phase mean)).sum) =
      (fun i => ((sampleRecurrence s X D B start count samples i).map (contribution phase mean)).sum) := by
  induction count generalizing start samples with
  | zero => rfl
  | succ count ih =>
    simp only [recurrence,sampleRecurrence]
    have ha := funext (advance_real rest phase s X D B start mean samples)
    rw [ha,ih]

theorem tree_real {B : Nat} (rest : Algebra ℝ) (values : Fin B → ℝ) (tree : ReductionTree B) :
    treeValue (ideal rest) values tree = (tree.leaves.map values).sum := by
  induction tree with
  | leaf i => change values i = ([values i]).sum; simp
  | node l r il ir =>
    change treeValue (ideal rest) values l + treeValue (ideal rest) values r =
      (List.map values (l.leaves ++ r.leaves)).sum
    simp only [il,ir,List.map_append,List.sum_append]

theorem flatMap_empty {α β : Type} (xs : List α) : xs.flatMap (fun _ => ([] : List β))=[] := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa only [List.flatMap_cons,List.nil_append] using ih

theorem sum_flatmap {α : Type} (xs : List α) (f : α→List ℝ) :
    (xs.flatMap f).sum=(xs.map (fun a => (f a).sum)).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa only [List.flatMap_cons,List.map_cons,List.sum_cons,List.sum_append] using congrArg ((f x).sum + ·) ih

theorem plain_sum_real (rest : Algebra ℝ) (phase : Bool) (s : State ℝ)
    (X : RegionName) (D B : Nat) (mean : ℝ) (tree : ReductionTree B)
    (hB : 0<B) (ht : tree.Complete) :
    treeValue (ideal rest) (allLanes (ideal rest) phase s X D B mean) tree =
      ((List.range D).map (fun j => contribution phase mean
        ((s.mem X (j+D*s.pids 0)).read .real))).sum := by
  let histories := sampleRecurrence s X D B 0 (blockCount D B) (fun _ => [])
  have hr := recurrence_real rest phase s X D B 0 (blockCount D B) mean (fun _ => [])
  have hp := recurrence_samples_perm s X D B 0 (blockCount D B) (fun _ => [])
  have hc : VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole.collectedSamples s X D B 0 (blockCount D B) =
      (List.range D).map (fun j => (s.mem X (j+D*s.pids 0)).read .real) := by
    rw [collected_samples_range]
    simpa [List.range_eq_range',Nat.min_eq_left (block_count_covers D B hB)] using
      emitted_range (fun j => (s.mem X (j+D*s.pids 0)).read .real) D (blockCount D B*B)
  have hperm : ((List.finRange B).flatMap histories).Perm
      ((List.range D).map (fun j => (s.mem X (j+D*s.pids 0)).read .real)) := by
    simpa [histories,hc,flatMap_empty] using hp
  have htotal := (hperm.map (contribution phase mean)).sum_eq
  rw [tree_real]
  have he : allLanes (ideal rest) phase s X D B mean =
      fun i => ((histories i).map (contribution phase mean)).sum := by
    simpa [allLanes,zero,ideal,histories] using hr
  rw [he,ht.map _ |>.sum_eq]
  simpa [List.map_flatMap,sum_flatmap,List.map_map,Function.comp_def] using htotal

/-- Independent mathematical specification over logical columns, with no chunk or tree. -/
def targetMean (s : State ℝ) (X : RegionName) (D : Nat) : ℝ :=
  ((List.range D).map (fun j => (s.mem X (j+D*s.pids 0)).read .real)).sum / D

def targetVariance (s : State ℝ) (X : RegionName) (D : Nat) : ℝ :=
  ((List.range D).map (fun j => ((s.mem X (j+D*s.pids 0)).read .real-targetMean s X D)^2)).sum / D

def targetInverse (s : State ℝ) (X : RegionName) (D : Nat) : ℝ :=
  1 / Real.sqrt (targetVariance s X D + 1e-05)

theorem mean_real (rest : Algebra ℝ) (s : State ℝ) (X : RegionName) (D B : Nat)
    (tree : ReductionTree B) (hB : 0<B) (ht : tree.Complete) :
    meanValue (ideal rest) s X D B tree = targetMean s X D := by
  unfold meanValue
  rw [plain_sum_real rest Bool.false s X D B _ tree hB ht]
  rfl

theorem inverse_real (rest : Algebra ℝ) (s : State ℝ) (X : RegionName) (D B : Nat)
    (tree : ReductionTree B) (hB : 0<B) (ht : tree.Complete) :
    inverseValue (ideal rest) s X D B tree = targetInverse s X D := by
  unfold inverseValue
  rw [mean_real rest s X D B tree hB ht,plain_sum_real rest Bool.true s X D B _ tree hB ht]
  rfl

#axiomsClean plain_sum_real
#axiomsClean mean_real
#axiomsClean inverse_real
end
end VeriTile.Bench.Repairs.LayerNormWelfold
