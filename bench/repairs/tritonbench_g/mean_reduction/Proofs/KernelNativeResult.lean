import bench.repairs.tritonbench_g.mean_reduction.Proofs.KernelFP
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.MeanReduction.KernelNativeResult
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open VeriTile.Bench.TritonBenchG.MeanReduction
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited VeriTile.Torch.FP32.Word := ⟨FP.Value.zero .fp32⟩

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (M N : Nat) : Program where
  kernel := kernel M N
  domain := []

def indexCommuted (M N : Nat) : Program :=
  { original M N with kernel := indexCommutedKernel M N }

/-- Only exact integer multiplication is commuted. Both kernels succeed and
 agree on complete memory for every floating backend and every input word. -/
theorem mean_index_equiv (M N : Nat) :
    original M N ≡[assumptions] indexCommuted M N := by
  apply VeriTile.Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ A D _ s _
  obtain ⟨t, he, hc⟩ := KernelFP.variants_execute (FP.ProfiledRewrite.engine A) M N s
  exact ⟨t,t,he,hc,rfl⟩

/-- Complete source and exact index-commuted source executions over raw FP32
cells, preserving the selected per-program primitive graph and full frame. -/
theorem mean_grid_fp32 (family : Nat → Algebra (FP.Value .fp32))
    (M N : Nat) (s : State (FP.Value .fp32)) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (Whole.gridSize M))) :
    ∃ t, WholeFPFamily.runTiles family Bool.false "X" "Mean" M N tiles s = some t ∧
      WholeFPFamily.runTiles family Bool.true "X" "Mean" M N tiles s = some t ∧
      (∀ i, i < M → t.mem "Mean" i =
        .mk .real (WholeFPFamily.rowValue family "X" M N s.mem i)) ∧
      (∀ (r : RegionName) o, r ≠ "Mean" ∨ M ≤ o → t.mem r o = s.mem r o) :=
  KernelFP.grid family M N s tiles schedule

#print_fp_assumptions mean_index_equiv
#axiomsClean mean_index_equiv
#axiomsClean mean_grid_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.KernelNativeResult
