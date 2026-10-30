import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.RawGridProof

namespace VeriTile.Bench.Repairs.Int4Matmul.RawCorrect
open VeriTile.Triton KernelExecution
noncomputable section

/-- Every source program succeeds. Its source-derived lane serialization gives
the independent dense INT4 sum at every output and preserves the full exterior.
The zeroed output is the original pre-kernel boundary for split atomics. -/
theorem matmul_grid_real (x : Raw.Arguments) (geometry : Raw.Geometry x)
    (tile : Tile) (member : tile∈Host.sourceTiles) (rest : FP.Structural.Algebra ℝ)
    (initial : ScalarState ℝ) (scratch : RegisterState ℝ) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes x.shape tile)) :
    (∀ l∈lanes, ∃ final, KernelExecution.exec realOps rest (Raw.parameters x tile)
      (atProgram (attach (zeroOutput x.shape realOps initial) scratch) l.pid l.split)=some final) ∧
    (∀ row col,row<x.shape.m → col<x.shape.n →
      (Raw.launch x tile realOps rest initial scratch lanes).mem "out" (cAddress x.shape row col)=
        FP.Structural.Cell.mk .real (mathematical x.shape initial row col)) ∧
    (∀ (r : RegionName) (i : Nat), r ≠ "out" ∨ x.shape.m * x.shape.n ≤ i →
      (Raw.launch x tile realOps rest initial scratch lanes).mem r i=(attach initial scratch).mem r i) := by
  refine ⟨?_,?_,?_⟩
  · intro l _
    obtain ⟨final,run,_⟩ := source_program_events realOps rest (Raw.parameters x tile)
      (atProgram (attach (zeroOutput x.shape realOps initial) scratch) l.pid l.split)
    exact ⟨final,run⟩
  · exact Raw.launch_real x geometry tile member rest initial scratch lanes complete
  · exact Raw.launch_frame x geometry tile member realOps rest initial scratch lanes complete

#axiomsClean matmul_grid_real
#specNonCircular mathematical avoiding [KernelSource.kernel,Raw.launch,parallel,computed]
end
end VeriTile.Bench.Repairs.Int4Matmul.RawCorrect
