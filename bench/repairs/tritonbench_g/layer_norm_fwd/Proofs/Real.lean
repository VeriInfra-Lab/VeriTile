import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.SourceBinding
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.LogicalInputs
import bench.tritonbench_g.layer_norm_fwd.Proofs.WholeFlatGrid
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WholeReduction

namespace VeriTile.Bench.Repairs.LayerNormFwd.RealProof
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
noncomputable section

/-- Reuse complete successful execution, every row, all eight returns and full
frame. No desired output, successful execution or trace-safety premise occurs. -/
theorem grid_return (q : HostReturn.Request) (r : Regions) (stride : Strides)
    (M N : Nat) (eps pHost p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ region, region ∉ A.regions → A.extent region = 0)
    (initial : BlockState) (hu : initial.undef = (fun _ _ => 0)) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated (HostReturn.flags q pHost) r)
    (hg : WholeGrid.Geometry r stride N)
    (hb : FlatDomain.Extents (HostReturn.flags q pHost) r stride M N A.extent) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
        (A.flattenKernel ((repaired (HostReturn.flags q pHost) r stride M N eps p block hBlock).eraseDType.toAlgKernel))
        (WholeGrid.grid M) (A.flattenState initial) (A.flattenState final)) ∧
      WholeGrid.Post (HostReturn.flags q pHost) r stride M N eps p initial final ∧
      HostReturn.observe (HostReturn.flags q pHost) r stride M N final =
        HostReturn.oracle (HostReturn.flags q pHost) r stride M N eps p initial :=
  WholeFlatGrid.original_correct (HostReturn.flags q pHost) r stride M N eps p block hBlock
    hN hsep hg A hd hc hb initial hu
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealProof
