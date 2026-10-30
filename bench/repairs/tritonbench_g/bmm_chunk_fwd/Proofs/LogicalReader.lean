import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.RealInterface
import bench.repairs.tritonbench_g.bmm_chunk_fwd.IO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem tensors_equiv (p : Parameters) (vals : RealInputs p) :
    (logicalInputsEquiv p vals).tensors = logicalInputs p vals := by
  change logicalInputs p ((envelopesEquiv p).symm (envelopesEquiv p vals)) = logicalInputs p vals
  rw [Equiv.symm_apply_apply]

theorem logical_iff (p : Parameters) (c : Host.Config) :
    (logicalIO p c ⊨ fun xs cell =>
      if p.causal=Bool.true ∧ cell.2.2.2.1.val<cell.2.2.2.2.val then (0 : ℝ) else
        Host.expected (parameters p) xs.tensors cell.1.val cell.2.1.val cell.2.2.1.val
          cell.2.2.2.1.val cell.2.2.2.2.val) ↔
    (realIO p c).Implements (realFormula p) := by
  change (realIO p c).Implements
    (fun vals (_ : Fin 1) (i : Fin p.shape.outputSize) =>
      let cell := (coordinateIndex p).symm i
      if p.causal=Bool.true ∧ cell.2.2.2.1.val<cell.2.2.2.2.val then (0 : ℝ) else
        Host.expected (parameters p) (logicalInputsEquiv p vals).tensors
          cell.1.val cell.2.1.val cell.2.2.1.val cell.2.2.2.1.val cell.2.2.2.2.val) ↔ _
  simp only [tensors_equiv]
  rfl

theorem logical_implements (p : Parameters) (c : Host.Config) (hc : c ∈ Host.configurations) :
    logicalIO p c ⊨ fun xs cell =>
      if p.causal=Bool.true ∧ cell.2.2.2.1.val<cell.2.2.2.2.val then (0 : ℝ) else
        Host.expected (parameters p) xs.tensors cell.1.val cell.2.1.val cell.2.2.1.val
          cell.2.2.2.1.val cell.2.2.2.2.val :=
  (logical_iff p c).mpr (real_implements p c hc)

#axiomsClean logical_iff
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO
