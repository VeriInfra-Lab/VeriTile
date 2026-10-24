import bench.repairs.tritonbench_g.add_example.Correct
import VeriTile.Torch.FP32

/-! Optional Tensor applications and Python-check correspondence. -/
namespace VeriTile.Bench.Repairs.AddExample.Contiguous.TensorBridge
open VeriTile.Triton
noncomputable section

open VeriTile.Torch.FP32 in
/-- Independent input binding through the same physical reader. Tensor API laws
and the checked metadata imply both logical reads. No storage inequality or
read-only disjointness premise is imposed; offset and overlapping views remain
admitted. These are entry observations, not assumptions about kernel results. -/
theorem checked_tensor_reads (read : Nat → Word) (x y : Tensor)
    (lx : TensorLaws x) (ly : TensorLaws y)
    (h : InputContract x.metadata y.metadata) (i : Nat) (hi : i < x.metadata.numel) :
    read ((dataView x lx h.contiguous.1).address i) = logicalValue read x lx i ∧
    read ((dataView y ly h.contiguous.2).address i) = logicalValue read y ly i := by
  refine ⟨direct_read_eq_logical read x lx h.contiguous.1 h.noNeg.1 i hi, ?_⟩
  exact direct_read_eq_logical read y ly h.contiguous.2 h.noNeg.2 i
    (by rw [← same_numel x.metadata y.metadata h]; exact hi)

open VeriTile.Torch.FP32 in
/-- The views used by `checked_tensor_reads` retain the actual launch pointers. -/
theorem checked_tensor_pointers (x y : Tensor) (lx : TensorLaws x) (ly : TensorLaws y)
    (h : InputContract x.metadata y.metadata) (hne : 0 < x.metadata.numel) :
    BitVec.ofNat 64 ((dataView x lx h.contiguous.1).address 0) = x.dataPtr ∧
    BitVec.ofNat 64 ((dataView y ly h.contiguous.2).address 0) = y.dataPtr := by
  refine ⟨actual_pointer x lx h.contiguous.1 hne, ?_⟩
  exact actual_pointer y ly h.contiguous.2
    (by rw [← same_numel x.metadata y.metadata h]; exact hne)

theorem python_checks_iff_preconditions (x y : ArgumentMetadata) :
    checkInputs x y = .ok () ↔ AdmittedArguments x y := checkInputs_iff x y
#axiomsClean checked_tensor_reads
#axiomsClean checked_tensor_pointers
#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddExample.Contiguous.TensorBridge
