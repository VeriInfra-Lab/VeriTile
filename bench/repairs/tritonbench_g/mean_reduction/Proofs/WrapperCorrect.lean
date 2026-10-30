import bench.repairs.tritonbench_g.mean_reduction.Kernels
import bench.repairs.tritonbench_g.mean_reduction.Proofs.Host

/-! Independent mathematical wrapper contract. The target reads the original
Tensor coordinates before the source's intentional compression allocation. -/
namespace VeriTile.Bench.Repairs.MeanReduction.Correctness
open VeriTile.Bench.TritonBenchG.MeanReduction Whole
noncomputable section

specification mean_dim_repaired_correct {x : Tensor rank} {dims : List (Fin rank)}
    {nodup : dims.Nodup} {entry : Nat → ℝ} {preexisting : Nat → Prop}
    (binding : TensorBinding metadata x)
    (h : Admitted metadata x.shape axes dtype dims)
    (rt : Runtime ℝ x dims nodup entry preexisting) (keepdim : Bool)
    (storage : ∀ c a, rt.compress=some c → rt.allocate c=some a → StorageBounds a) :
    metadata.shape=List.ofFn x.shape ∧
    (∃ result, runReal metadata axes dtype rt keepdim=.ok result ∧
      HostOutcomes (fun row=>mathematicalMean x.shape dims (x.read entry)
        ((batchLinear x.shape dims).symm row)) rt keepdim result) ∧
    ∀ c a, rt.compress=some c → rt.allocate c=some a →
      ∀ pid t lane col, pid<Whole.gridSize (rows x.shape dims) →
        t<Whole.gridSize (columns x.shape dims) → lane<8 → col<8 →
        pid*8+lane<rows x.shape dims → t*8+col<columns x.shape dims →
        (nativeInputPointer c.base pid lane (columns x.shape dims) (t*8) col).toNat=
          4*(c.base+(pid*8+lane)*columns x.shape dims+(t*8+col)) ∧
        (nativeOutputPointer a.base pid lane).toNat=4*(a.base+pid*8+lane) := by
  exact ⟨binding.shape,real_outcomes h rt keepdim,fun c a hc ha=>
    active_pointers a h.2 (storage c a hc ha)⟩

#axiomsClean mean_dim_repaired_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.Correctness
