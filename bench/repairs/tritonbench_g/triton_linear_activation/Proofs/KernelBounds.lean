import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.KernelGeometry

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelBounds
open VeriTile.Triton FP.Structural KernelSource KernelValues KernelContract
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem positive_dimensions (r : Request) (c : Fin 98) (pid : Nat) (valid : pid<r.shape.programs c) :
    0<r.batch.rows ∧ 0<r.n := by
  have cb := config_bounds c
  have bm : 0<(configs c).bm := by omega
  have bn : 0<(configs c).bn := by omega
  constructor
  · by_contra h
    have empty : r.batch.rows=0 := by omega
    simpa [Shape.programs,Shape.rows,Request.shape,empty,ceil_zero _ bm] using valid
  · by_contra h
    have empty : r.n=0 := by omega
    simpa [Shape.programs,Shape.cols,Request.shape,empty,ceil_zero _ bn] using valid

/-- All formed A/B pointer offsets, including masked lanes and the final
post-loop increment, fit the existing signed-32 source guard. There is no
additional stride, shape, or numerical guard. -/
theorem formed_inputs (r : Request) (c : Fin 98) (regions : Fin 5 → RegionName)
    (s : State α) (ts : Fin 4 → Checked.Observed) (flag : Bool)
    (admit : Checked.Domain r ts flag) (pid : s.pids 0<r.shape.programs c)
    (cur : Nat) (stepBound : cur≤ceilDiv r.k (configs c).bk)
    (a : TileIndex [(configs c).bm,(configs c).bk]) (b : TileIndex [(configs c).bk,(configs c).bn]) :
    aOffset (parameters r c regions) s cur a<2^31 ∧
      bOffset (parameters r c regions) s cur b<2^31 := by
  have dims := positive_dimensions r c (s.pids 0) pid
  have cb := config_bounds c
  have bk : 0<(configs c).bk := by omega
  have ax := formed_input r.batch.rows r.k (configs c).bk a.2.1.val
    (KernelValues.row (parameters r c regions) s a.1%r.batch.rows) cur bk (by omega) a.2.1.isLt
    (Nat.mod_lt _ dims.1) stepBound admit.bounds.2.2.1
  have bx := formed_input r.n r.k (configs c).bk b.1.val
    (KernelValues.col (parameters r c regions) s b.2.1%r.n) cur bk (by omega) b.1.isLt
    (Nat.mod_lt _ dims.2) stepBound admit.bounds.2.2.2
  constructor
  · simpa [aOffset,parameters,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using ax
  · simpa [bOffset,parameters,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using bx

#axiomsClean formed_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelBounds
