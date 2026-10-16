import bench.tritonbench_g.triton_softmax.Kernels
import bench.tritonbench_g.triton_softmax.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

/-- No numerical or domain assumptions: the candidate changes only integer
multiplication in the input row address. -/
def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) : Program where
  kernel := typedKernel input output v out inp si so n B
  domain := []

/-- Equality of complete typed execution, including registers, all memory
cells and failure. Negative-infinity padding, both masks, max/sub/exp/sum/div,
all source precisions and the final destination conversion remain unchanged. -/
specification softmax_index_equiv (input output : Storage) (out inp : RegionName)
    (si so n B : Nat) :
    program input output Bool.false out inp si so n B ≡[assumptions]
      program input output Bool.true out inp si so n B := by
  cases input <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions softmax_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax.FPEquiv
