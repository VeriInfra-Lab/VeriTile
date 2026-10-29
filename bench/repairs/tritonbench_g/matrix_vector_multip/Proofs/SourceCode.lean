import bench.repairs.tritonbench_g.matrix_vector_multip.Kernels
import VeriTile.Triton.Source.While

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourceCode
open VeriTile.Triton

variable (N M BN BM : Nat) (rowShape : Random.ArangeSizeDomain BN)
  (columnShape : Random.ArangeSizeDomain BM)

theorem original_has_while :
    (Source.whileLoop? (Kernels.source N M BN BM rowShape columnShape).surfaceBody).isSome = Bool.true := rfl

theorem candidate_has_while :
    (Source.whileLoop? (Kernels.explicitFMA N M BN BM rowShape columnShape).surfaceBody).isSome = Bool.true := rfl

def original : Source.WhileLoop :=
  (Source.whileLoop? (Kernels.source N M BN BM rowShape columnShape).surfaceBody).get
    (original_has_while N M BN BM rowShape columnShape)

def candidate : Source.WhileLoop :=
  (Source.whileLoop? (Kernels.explicitFMA N M BN BM rowShape columnShape).surfaceBody).get
    (candidate_has_while N M BN BM rowShape columnShape)

theorem original_decomposition :
    (Kernels.source N M BN BM rowShape columnShape).surfaceBody =
      (original N M BN BM rowShape columnShape).before ++
      [.whileLoop (original N M BN BM rowShape columnShape).condition
        (original N M BN BM rowShape columnShape).body] ++
      (original N M BN BM rowShape columnShape).after := rfl

theorem candidate_decomposition :
    (Kernels.explicitFMA N M BN BM rowShape columnShape).surfaceBody =
      (candidate N M BN BM rowShape columnShape).before ++
      [.whileLoop (candidate N M BN BM rowShape columnShape).condition
        (candidate N M BN BM rowShape columnShape).body] ++
      (candidate N M BN BM rowShape columnShape).after := rfl

theorem complete_statement_counts :
    (original N M BN BM rowShape columnShape).before.length = 14 ∧
    (original N M BN BM rowShape columnShape).body.length = 7 ∧
    (original N M BN BM rowShape columnShape).after.length = 3 := ⟨rfl,rfl,rfl⟩


end VeriTile.Bench.Repairs.MatrixVectorMultip.SourceCode
