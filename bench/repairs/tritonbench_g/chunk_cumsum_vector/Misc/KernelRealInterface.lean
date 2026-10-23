import bench.repairs.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Contract
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.RealStorage
import VeriTile.Meta.Specification

/-! One actual program computes inclusive prefixes for its feature tile,
with explicit zero padding, independent initial-memory values and the full
physical exterior frame. Only the Real interpretation erases dtype. -/
namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealInterface
open VeriTile.Triton
noncomputable section

abbrev source := Kernels.source

theorem memory_correct (h : Call) (layout : Geometry.OutputLayout h)
    (p : Storage.Placement h) (padding : Config → Geometry.Program h → RegionName → Nat → ℝ)
    (c : Config) (chosen : c ∈ configurations) (program : Geometry.Program h)
    (before : RealStorage.Heap) :
    Spec.Real (∃ after, RealStorage.execute h p padding c program before = some after ∧
      Stream.StableInput h p before after ∧
      (∀ time j, Grid.active h program time j →
        RealStorage.read (after (Storage.address p.backing.output (Grid.offset h program time j))) =
          mathematical (fun k => RealStorage.read (before (Storage.address p.backing.input
            (h.inputStrides.address h.H program.row k (program.feature*BS+j.val))))) h.T time) ∧
      (∀ addr, ¬Storage.writes h p [program] addr → after addr = before addr)) :=
  RealStorage.complete h layout p padding c chosen program before

#specNonCircular mathematical avoiding [Kernels.source,Kernels.sumLeft]
#axiomsClean memory_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealInterface
