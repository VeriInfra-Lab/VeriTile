import bench.tritonbench_g.dequantize_rowwise.Proofs.FPExecution
import VeriTile.Triton.Float.ProfiledMemory

/-! The only transformation commutes the exact integer row-start product.
INT8 loads, FP32 multiplication order, scalar precision and FP16 conversion
are unchanged. Both executions succeed and their complete typed memories
agree. No finite-operand restriction or numerical experiment is needed. -/
namespace VeriTile.Bench.Repairs.DequantizeRowwise.FPEquivalence
open VeriTile VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
/-- Selected INT8/FP32/FP16 source with the production reciprocal and ports. -/
def selectedOriginal (p : Parameters) : Program := ⟨kernel p, []⟩
/-- The same typed source with only the integer row-base product commuted. -/
def selectedCandidate (p : Parameters) : Program := ⟨indexCommutedKernel p, []⟩

/-- The wrapper's actual fixed reciprocal, ports and row geometry. -/
specification selected_equivalent (p : Parameters) :
    selectedOriginal p ≡[assumptions] selectedCandidate p :=
  FPExecution.equivalence xRegion scaleRegion outRegion (1 / 127)
    (p.rows * p.cols) p.cols p.width

#print_fp_assumptions selected_equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FPEquivalence
