import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.GemmPairIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIndices
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ResetPairIO

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndicesIO
open VeriTile Triton FP.Structural KernelExecution
noncomputable section
variable {F I H : Type} [Inhabited F]

theorem original_source (rowScale output : Precision) (p : GemmIndices.Parameters) :
    Gemm.source rowScale output p=GemmIndices.original rowScale output p := by
  cases rowScale <;> cases output <;> rfl

theorem invocation_run (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : PublicIO.Gemm.Program s c) (base : State F) :
    (PublicIO.Gemm.invocation M carrier ops zero plans rowScale output s c columnMajor p).run base=
      (invocation M carrier ops zero plans rowScale output s c columnMajor p).run base := by
  unfold IntegerDotInvocation.run
  change IntegerDot.exec M _ _ (Gemm.source rowScale output _) _=IntegerDot.exec M _ _ _ _
  rw [original_source]
  exact GemmIndices.execution M _ _ rowScale output _ _
    (PublicIO.Gemm.rows_pos s c p.1 p.1.isLt) (PublicIO.Gemm.columns_pos s c p.1 p.1.isLt)

theorem invocation_safe (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (zero : carrier.decode ops.izero=0) (plans : Plans) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (p : PublicIO.Gemm.Program s c)
    (bounds : RegionName→Nat) (base : State F) :
    (PublicIO.Gemm.invocation M carrier ops zero plans rowScale output s c columnMajor p).Safe bounds base ↔
      (invocation M carrier ops zero plans rowScale output s c columnMajor p).Safe bounds base := by
  unfold IntegerDotInvocation.Safe
  change IntegerDot.TraceSafe M _ _ bounds (Gemm.source rowScale output _) _ ↔
    IntegerDot.TraceSafe M _ _ bounds _ _
  rw [original_source]
  exact GemmIndices.safety M _ _ rowScale output _ bounds _
    (PublicIO.Gemm.rows_pos s c p.1 p.1.isLt) (PublicIO.Gemm.columns_pos s c p.1 p.1.isLt)

local instance : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

theorem equivalent (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) :
    ResetPairIO.EquivalentResetOnCells
      (PublicIO.Gemm.nativeIO f tree rest plans rowScale output s c columnMajor q)
      (variant (KernelScalarProfile.native f tree rest) signed32 (KernelNativeModel.operations f).toOps
        rfl plans rowScale output s c columnMajor q)
      (PublicIO.Gemm.SignedInputs s rowScale) (PublicIO.Gemm.initialValues (FP.Value.zero .fp32) s output) := by
  apply ResetPairIO.of_run_safe _ rfl
  · intro xs event base
    exact invocation_run _ _ _ _ _ _ _ _ _ _ event base
  · intro xs event bounds base
    exact invocation_safe _ _ _ _ _ _ _ _ _ _ event bounds base
  · exact PublicIO.Gemm.native_implements f tree rest plans rowScale output s c columnMajor q

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.GemmIndicesIO
