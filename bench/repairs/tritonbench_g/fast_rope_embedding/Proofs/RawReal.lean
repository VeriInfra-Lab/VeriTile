import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.RawReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding KernelInterface
noncomputable section

/-- Complete mathematical source execution, all rotated outputs and the full
unwritten-cell frame over raw state, geometry and the source stride arguments. -/
theorem fast_rope_raw_correct (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (s : State ℝ) (row group : Nat) (launch : r.Launched row group)
    (strideEq : row*qStride=row*(r.heads*r.dimension))
    (cosineEq : cosineStride=r.half) (sineEq : sineStride=r.half) :
    ∃ final, FP.Structural.exec ideal.backend (rawSource r qStride cosineStride sineStride backward)
        (programState s row group)=some final ∧
      Kernel.Post r row group s final (fun h col=>Spec.result r backward
        (fun o=>(s.mem "Q" o).read .real) (fun o=>(s.mem "C" o).read .real)
        (fun o=>(s.mem "S" o).read .real) row h col) := by
  simpa only [ideal_result] using
    KernelInterface.executes ideal r qStride cosineStride sineStride backward s row group
      launch strideEq cosineEq sineEq

/-- Every rotated coordinate is covered, including padded blocks and
partial head groups. This describes the kernel grid, not host launch code. -/
theorem fast_rope_raw_coverage (r : Request) (row head col : Nat)
    (hr : row<r.rows) (hh : head<r.heads) (hc : col<2*r.half) :
    ∃ group,r.Launched row group ∧
      Kernel.Written r row group "Q" (r.offset row head col) := by
  obtain ⟨group,hg,written⟩ := Kernel.complete_coverage r row head col hr hh hc
  exact ⟨group,⟨by omega,hr,hg⟩,written⟩

/-- Empty payloads and D<2 have an empty declared kernel grid. -/
theorem fast_rope_raw_no_launch (r : Request) (empty : r.size=0 ∨ r.dimension<2) :
    ∀ row group,¬r.Launched row group := by
  intro row group launch
  have head := group_head r group launch.2.2
  have half := launch.1
  have rows := launch.2.1
  rcases empty with empty | empty
  · unfold Request.size at empty
    have dim : 0<r.dimension := by unfold Request.half at half; omega
    have positive := Nat.mul_pos (Nat.mul_pos (show 0<r.rows by omega)
      (show 0<r.heads by omega)) dim
    omega
  · unfold Request.half at half; omega

#axiomsClean fast_rope_raw_correct
#axiomsClean fast_rope_raw_coverage
#axiomsClean fast_rope_raw_no_launch
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.RawReal
