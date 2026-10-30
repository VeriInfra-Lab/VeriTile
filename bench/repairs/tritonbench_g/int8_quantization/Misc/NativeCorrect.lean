/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.int8_quantization.Proofs.NativeInterface

namespace VeriTile.Bench.Repairs.Int8Quantization.FloatingSpecification
open VeriTile.Triton FP.Structural Revision002 Words
noncomputable section
set_option autoImplicit false

/-- The complete source grid with the original scalar primitive model.
All FP32 words and the original instruction/reduction choices remain visible. -/
def quantizationIO (M : Algebra Word) (ops : Model.Primitives Word) (p : Parameters) :=
  QuantizationIO.quantizationIO M ops p

def agreements := @QuantizationIO.Agreements Word
def reference := @QuantizationIO.softwareTarget Word

/-- The most general existing full-grid source/software relation. Its domain
is exactly the two per-group context laws and fourteen reached primitive
agreements; source access safety, execution and all outputs/frame are proved.
The ordered native specialization derives these agreements directly in
Proofs/NativeInterface.lean and needs no numerical experiment. -/
theorem per_block_int8_equiv (p : Parameters) (positive : 0 < p.width)
    (block : 0 < p.blockRows) (M : Algebra Word) (ops : Model.Primitives Word)
    (software : Floating.Numerics Word) :
    (quantizationIO M ops p).ImplementsOn (agreements M ops software p) (reference M software p) :=
  QuantizationIO.software_implements p positive block M ops software

#stmtSurfaceSubset per_block_int8_equiv ⊆
  [quantizationIO, agreements, reference, ProgramIO.ImplementsOn, Parameters, Parameters.width,
   Parameters.blockRows, FP.Structural.Algebra, Word, Model.Primitives, Floating.Numerics, Byte]
#specNonCircular reference avoiding [source,FP.Structural.exec,Grid.model,quantized,savedScale]
#axiomsClean per_block_int8_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8Quantization.FloatingSpecification
