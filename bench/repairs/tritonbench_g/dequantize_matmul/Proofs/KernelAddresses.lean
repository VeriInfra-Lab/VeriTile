import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelAddresses
open VeriTile.Triton KernelSource KernelResult KernelTensor
noncomputable section

def lane (a : Direct.TensorInfo) (b : Int8Tensor) (config : LaunchConfig)
    (member : config ∈ launchConfigs) (pid : Program) (i : KernelResult.Lane (parameters a b config)) :
    VeriTile.Bench.Repairs.DequantizeMatmul.Lane (config.tile member) :=
  ⟨pid.1, pid.2, i.1, i.2.1⟩

/-- Native signed64 coordinate/address arithmetic for every actual launch
and tail lane is inherited from the existing guard-derived IndexABI proof.
No numerical value or result is a premise. -/
def Exact (p : Parameters) : Prop :=
  ∀ pid, pid ∈ KernelGrid.order p → ∀ i : KernelResult.Lane p,
    (coordinate64 pid.1 p.BK i.1.val).toInt = (row p pid i : Int) ∧
    (coordinate64 pid.2 p.BN i.2.1.val).toInt = (col p pid i : Int) ∧
    (affine64 (row p pid i) (col p pid i) p.N 1).toInt = (offset p pid i : Int) ∧
    (BitVec.ofNat 64 (col p pid i)).toInt = (col p pid i : Int) ∧
    4 * offset p pid i < 2^63 ∧ 4 * col p pid i < 2^63

theorem exact (a : Direct.TensorInfo) (b : Int8Tensor) (scale : Tensor)
    (out : Option Direct.TensorInfo) (h : Direct.InputContract a (int8Info b) (fp32Info scale) out)
    (config : LaunchConfig) (member : config ∈ launchConfigs) : Exact (parameters a b config) := by
  intro pid hp i
  have pos := launchConfigs_positive config member
  have hp' : (pid.1, pid.2) ∈ KernelGrid.order (parameters a b config) := hp
  have both : pid.1 ∈ List.range (KernelGrid.gridCount (Direct.columns a) config.blockK) ∧
      pid.2 ∈ List.range (KernelGrid.gridCount (Direct.columns (int8Info b)) config.blockN) :=
    List.mem_product.mp hp'
  have launch : pid.1 < gridCount (Direct.columns a) config.blockK ∧
      pid.2 < gridCount (Direct.columns (int8Info b)) config.blockN := by
    simpa [KernelGrid.gridCount, gridCount] using both
  have hK : 0 < Direct.columns a := by
    by_contra hh
    have hz : Direct.columns a = 0 := by omega
    have empty : gridCount (Direct.columns a) config.blockK = 0 := by
      rw [hz]
      exact Nat.div_eq_of_lt (by simpa using Nat.sub_lt pos.1 (by decide : 0 < 1))
    rw [empty] at launch
    omega
  have hN : 0 < Direct.columns (int8Info b) := by
    by_contra hh
    have hz : Direct.columns (int8Info b) = 0 := by omega
    have empty : gridCount (Direct.columns (int8Info b)) config.blockN = 0 := by
      rw [hz]
      exact Nat.div_eq_of_lt (by simpa using Nat.sub_lt pos.2 (by decide : 0 < 1))
    rw [empty] at launch
    omega
  have admitted := Direct.native_admitted a (int8Info b) (fp32Info scale) out h hK hN config member
  have launched : (lane a b config member pid i).launched (Direct.shape a (int8Info b)) := launch
  have native := admitted.exact (lane a b config member pid i) launched
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact native.1
  · exact native.2.1
  · simpa [lane, parameters, Direct.canonical2, offset,
      VeriTile.Bench.Repairs.DequantizeMatmul.Lane.k,
      VeriTile.Bench.Repairs.DequantizeMatmul.Lane.n, row, col] using native.2.2.1
  · simpa [lane, parameters, Direct.canonical1,
      VeriTile.Bench.Repairs.DequantizeMatmul.Lane.n, col] using native.2.2.2.2.1
  · simpa [lane, parameters, Direct.canonical2, offset,
      VeriTile.Bench.Repairs.DequantizeMatmul.Lane.k,
      VeriTile.Bench.Repairs.DequantizeMatmul.Lane.n, row, col] using native.2.2.2.2.2.1
  · simpa [lane, parameters, Direct.canonical1,
      VeriTile.Bench.Repairs.DequantizeMatmul.Lane.n, col] using native.2.2.2.2.2.2

#axiomsClean exact
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelAddresses
