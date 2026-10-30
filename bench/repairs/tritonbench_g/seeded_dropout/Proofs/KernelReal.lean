import bench.repairs.tritonbench_g.seeded_dropout.Proofs.KernelExecution
import bench.repairs.tritonbench_g.seeded_dropout.Proofs.KernelMemory

namespace VeriTile.Bench.Repairs.SeededDropout.KernelReal
open VeriTile.Triton FP.Structural KernelExecution
open TritonBenchG.SeededDropout
noncomputable section

/-- Successful execution of the displayed FP32 source in its Real interpretation.
Canonical Philox10 and its rounded random decoder are retained. The surrounding
Real arithmetic does not claim native FP32 exactness. -/
theorem seeded_dropout_kernel_correctness (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) (s : State ℝ)
    (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed)
    (capacity : RegionBounds) (bounds : KernelMemory.Bounds X O n capacity) :
    Spec.Real (
      (∀ lane : Fin 1024, pid*1024+lane.val < n →
        pid*1024+lane.val < capacity X ∧ pid*1024+lane.val < capacity O) ∧
      ∃ t, FP.Structural.exec Whole.realAlgebra (kernel k X O n p seed) (entry pid s) = some t ∧
        (∀ lane : Fin 1024, pid*1024+lane.val < n →
          t.mem O (pid*1024+lane.val) = Cell.mk .real
            (Whole.realFormula p (seedValue k seed) (pid*1024+lane.val)
              ((s.mem X (pid*1024+lane.val)).read .real))) ∧
        (∀ r a, Outside O n pid r a → t.mem r a = s.mem r a)) := by
  refine ⟨fun lane active => (KernelMemory.active_indices X O n pid capacity bounds hn hp lane).2.2 active, ?_⟩
  let result := originalExecution Whole.realAlgebra (fun a b => decide (a < b)) rfl
    k X O n p seed pid s hn hp hs
  refine ⟨result.val,result.property,?_,?_⟩
  · intro lane active
    simpa only [ActualTile.scalar_real] using
      values Whole.realAlgebra (fun a b => decide (a < b)) rfl k X O n p seed pid s hn hp hs lane active
  · exact frame Whole.realAlgebra (fun a b => decide (a < b)) rfl k X O n p seed pid s hn hp hs

#specNonCircular Whole.realFormula avoiding [kernel,indexCommutedKernel,originalExecution]
#axiomsClean seeded_dropout_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout.KernelReal
