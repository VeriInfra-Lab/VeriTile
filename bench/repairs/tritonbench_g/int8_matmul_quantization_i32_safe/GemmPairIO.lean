import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIndices
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ResetPairIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndicesIO
open VeriTile Triton FP.Structural KernelExecution
noncomputable section
variable {F I H : Type} [Inhabited F]

def invocation (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : PublicIO.Gemm.Program s c) : IntegerDotInvocation F :=
  { PublicIO.Gemm.invocation M carrier ops zero plans rowScale output s c columnMajor p with
    kernel:=GemmIndices.candidate rowScale output (Gemm.parameters s c columnMajor) }

def variant (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    ResetPairIO.Variant (PublicIO.Gemm.io M carrier ops zero plans rowScale output s c columnMajor q) where
  invocation:=fun _=>invocation M carrier ops zero plans rowScale output s c columnMajor
  observe:=fun _ _=>id

local instance : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

def original (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) :=
  ResetPairIO.Implementation.original
    (PublicIO.Gemm.nativeIO f tree rest plans rowScale output s c columnMajor q)
    (PublicIO.Gemm.SignedInputs s rowScale) (PublicIO.Gemm.initialValues (FP.Value.zero .fp32) s output)

def optimized (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) :=
  ResetPairIO.Implementation.ofVariant
    (variant (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps
      rfl plans rowScale output s c columnMajor q)
    (PublicIO.Gemm.SignedInputs s rowScale) (PublicIO.Gemm.initialValues (FP.Value.zero .fp32) s output)

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndicesIO
