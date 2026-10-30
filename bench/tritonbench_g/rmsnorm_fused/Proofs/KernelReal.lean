import bench.tritonbench_g.rmsnorm_fused.Proofs.KernelBinding
import bench.tritonbench_g.rms_norm_triton.Proofs.Mathematical_WholeMath

namespace VeriTile.Bench.Repairs.RmsnormFused.Ideal
open VeriTile Triton FP.Structural Binding
open VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole (realAlgebra realSum sum_prefix_any)
noncomputable section
set_option maxHeartbeats 5000000

/-- Mathematical RMSNorm on the actual Tensor's logical storage indices. -/
def result (x w : Tensor) (memory : Nat → ℝ) (N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  x.read memory k *
    (Real.sqrt ((∑ j ∈ Finset.range N,(x.read memory ((k/N)*N+j))^2)/(N:ℝ)+eps))⁻¹ *
    w.read memory (k%N)

theorem value_real (s : State ℝ) (N B row col : Nat) (eps : ℝ)
    (active : col<N) (covered : N≤B) :
    Runtime.value realAlgebra s "X" "W" row N B eps col =
      Runtime.read s "X" (row*N+col) *
        (Real.sqrt ((∑ j ∈ Finset.range N,Runtime.read s "X" (row*N+j)^2)/(N:ℝ)+eps))⁻¹ *
          Runtime.read s "W" col := by
  simp only [Runtime.value,Runtime.rstd,Runtime.variance,Runtime.square,Runtime.xload,Runtime.wload,
    Runtime.lit,Runtime.add,Runtime.mul,Runtime.div,Runtime.cast,realAlgebra,realSum,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,active,if_true,id_eq]
  norm_num only
  have terms : (fun j : Fin B =>
      0+(if j.val<N then (if j.val<N then Runtime.read s "X" (row*N+j.val) else 0) else 0)*
        (if j.val<N then (if j.val<N then Runtime.read s "X" (row*N+j.val) else 0) else 0)) =
      (fun j : Fin B => if j.val<N then Runtime.read s "X" (row*N+j.val)^2 else 0) := by
    funext j
    split_ifs <;> simp [pow_two]
  rw [terms]
  simp only [one_div]
  apply congrArg (fun total : ℝ => Runtime.read s "X" (row*N+col) *
    (Real.sqrt (total/(N:ℝ)+eps))⁻¹ * Runtime.read s "W" col)
  exact sum_prefix_any N B _ covered (fun j => Runtime.read s "X" (row*N+j)^2)

theorem logical_value (x w y : Tensor) (memory : Nat → ℝ) (eps : EpsilonInfo)
    (c : InputContract x.info w.info eps) (row col : Nat)
    (hr : row<(geometry x.info).rows) (hc : col<width x.info) :
    Runtime.value realAlgebra (initial x w y memory) "X" "W" row (width x.info)
      (RmsnormFused.block (width x.info)) eps.value col =
        result x w memory (width x.info) eps.value (row*width x.info+col) := by
  rw [value_real _ _ _ _ _ _ hc c.block]
  have size := input_sizes x w eps c
  have index : row*width x.info+col<x.size := by
    rw [size.1]
    exact offset_bounded (geometry x.info) row col hr hc
  have coordinates := offset_coordinates (geometry x.info) (geometry_valid x.info w.info eps c) row col hc
  simp only [Shape.offset,geometry] at coordinates
  unfold result
  rw [coordinates.1,coordinates.2,x.read_view memory _ index,w.read_view memory _ (by omega)]
  have sumEqual : (∑ j ∈ Finset.range (width x.info), Runtime.read (initial x w y memory) "X" (row*width x.info+j)^2) =
      (∑ j ∈ Finset.range (width x.info), x.read memory (row*width x.info+j)^2) := by
    apply Finset.sum_congr rfl
    intro j hj
    rw [x.read_view memory _ (by rw [size.1]; exact offset_bounded (geometry x.info) row j hr (Finset.mem_range.mp hj))]
    rfl
  rw [sumEqual]
  rfl

#guard_msgs (drop info) in
#specNonCircular result avoiding [source,Runtime.value,Runtime.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Ideal
