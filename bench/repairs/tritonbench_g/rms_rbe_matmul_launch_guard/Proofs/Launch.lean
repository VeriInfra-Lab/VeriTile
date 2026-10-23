import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Contract
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelBinding
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelGrid

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Launch
open RmsRbeMatmul
open RmsMatmulRbe.Contract (dim)
open RmsRbeMatmulLaunchGuard.Contract

theorem actual_grid (i : Inputs) : grid i =
    ((Checked.shape i).B, KernelGrid.tilesM (KernelBinding.parameters i) *
      KernelGrid.tilesN (KernelBinding.parameters i)) := rfl

/-- Each actual program coordinate is in the two admitted source axes.
There is deliberately no bound on the product of the axes. -/
theorem program_bounds (i : Inputs) (h : InputContract i) (q : KernelGrid.Program)
    (member : q ∈ KernelGrid.programs (Checked.shape i).B (KernelBinding.parameters i)) :
    q.1 < 2^31 ∧ q.2 < 65535 := by
  rcases q with ⟨batch, program⟩
  have bounds : batch < dim i.x 0 ∧ program < (grid i).2 := by
    simpa [KernelGrid.programs, KernelGrid.tilesM, KernelGrid.tilesN,
      KernelBinding.parameters, Checked.shape, grid] using member
  have rows : 0 < dim i.x 1 := by
    by_contra nonpositive
    have zero : dim i.x 1 = 0 := by omega
    simpa [grid, zero] using bounds.2
  have admitted := h.2 (by omega) rows
  change dim i.x 0 ≤ 2^31-1 ∧ (grid i).2 ≤ 65535 at admitted
  constructor <;> omega

theorem signed_pid (n : Nat) (bound : n < 2^31) :
    (BitVec.ofNat 32 n).toInt = (n : Int) := by
  have word : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt, word]
  rw [word]
  omega

theorem native_program_ids (i : Inputs) (h : InputContract i) (q : KernelGrid.Program)
    (member : q ∈ KernelGrid.programs (Checked.shape i).B (KernelBinding.parameters i)) :
    (BitVec.ofNat 32 q.1).toInt = (q.1 : Int) ∧
    (BitVec.ofNat 32 q.2).toInt = (q.2 : Int) := by
  have bounds := program_bounds i h q member
  exact ⟨signed_pid q.1 bounds.1, signed_pid q.2 (by omega)⟩

#axiomsClean actual_grid
#axiomsClean program_bounds
#axiomsClean native_program_ids
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Launch
