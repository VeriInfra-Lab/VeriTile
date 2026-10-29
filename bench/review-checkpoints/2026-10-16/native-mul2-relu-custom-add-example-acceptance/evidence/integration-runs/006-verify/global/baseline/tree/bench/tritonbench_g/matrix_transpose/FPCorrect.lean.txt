import bench.tritonbench_g.matrix_transpose.Proofs.FPSoftware
import bench.tritonbench_g.matrix_transpose.Proofs.FPWitness
import bench.tritonbench_g.matrix_transpose.Proofs.FPShape

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose
open VeriTile.Triton FP.Structural FPBits

/-- Complete fp16 bit-preserving transpose, for both already accepted source
versions, all original dimensions/strides and arbitrary initial memory. The
only premise is the original output-address injectivity. All loads precede
stores, so aliases are allowed. This is a software typed-memory result;
host allocation, native signed32 address lowering and GPU byte ABI are separate. -/
specification matrix_transpose_bits_correctness (v : Bool)
    (p : FPBits.Parameters) (s : State FPBits.Word)
    (hOutInj : Function.Injective (FPBits.outputOffset p)) :
    ∃ t, FP.Structural.exec FPBits.unusedPrimitives (FPBits.source v p) s = some t ∧
      t.mem = FPBits.referenceMemory p s ∧
      (∀ i : TileIndex [p.d,p.m], t.mem p.output (FPBits.outputOffset p i) =
        .mk .fp16 (FPBits.outputValue p s i)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.d,p.m], o ≠ FPBits.outputOffset p i) →
        t.mem r o = s.mem r o) :=
  FPBits.software_correctness v p s hOutInj

#axiomsClean matrix_transpose_bits_correctness
#specNonCircular FPBits.referenceMemory avoiding [FPBits.source, FPBits.execution, FP.Structural.exec]
end VeriTile.Bench.TritonBenchG.MatrixTranspose
