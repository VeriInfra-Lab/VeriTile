/- Auxiliary single-source native correctness. No program-pair equivalence is claimed. -/
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIOResults
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIOResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.FPEquiv
open VeriTile.Triton FP.Structural KernelExecution
open KernelProducerGrid (Geometry)
noncomputable section
local instance kernelInhabitedWord32 : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

/-- Exact native producer observation: all FP32 words, original reduction
selection, INT8 conversion and FP16 scale storage remain admitted. -/
theorem producer_native (f : KernelNativeModel.FloatingInstructions) (p : Geometry)
    (selection : Producer.Selection p) (covers : p.columns≤p.width) (rest : Algebra Word32)
    (order : Producer.Schedule p) :
    (PublicIO.Producer.nativeIO f p selection rest order).Implements
      (PublicIO.Producer.nativeFormula f p selection) :=
  PublicIO.Producer.native_implements f p selection covers rest order

/-- Exact native source observation, including signed INT32 accumulation,
FP32 scalar operations, mandatory FP16 partial, selected output precision and
each cell's original atomic order. The primitive instruction model is the
existing boundary; this asserts no floating rewrite or native=Real identity. -/
theorem gemm_native (f : KernelNativeModel.FloatingInstructions) (tree : KernelScalarProfile.Tree)
    (rest : Algebra Word32) (plans : Plans) (rowScale output : Precision) (s : Shape) (c : Config)
    (columnMajor : Bool) (q : Schedule s c) :
    (PublicIO.Gemm.nativeIO f tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells
      (PublicIO.Gemm.SignedInputs s rowScale) (PublicIO.Gemm.initialValues (FP.Value.zero .fp32) s output)
      (PublicIO.Gemm.nativeFormula f tree rest plans rowScale output s c columnMajor q) :=
  PublicIO.Gemm.native_implements f tree rest plans rowScale output s c columnMajor q

/- Both entries observe the original source under its retained primitive
model. No transformation law or deferred numerical experiment is introduced. -/
#stmtSurfaceSubset producer_native ⊆
  [KernelNativeModel.FloatingInstructions,Geometry,Geometry.columns,Geometry.width,
   Producer.Selection,Producer.Schedule,FP.Structural.Algebra,Word32,kernelInhabitedWord32,
   PublicIO.Producer.nativeIO,PublicIO.Producer.nativeFormula,ProgramIO.Implements]
#stmtSurfaceSubset gemm_native ⊆
  [KernelNativeModel.FloatingInstructions,KernelScalarProfile.Tree,FP.Structural.Algebra,Word32,kernelInhabitedWord32,
   Plans,Precision,Shape,Config,Schedule,PublicIO.Gemm.Program,PublicIO.Gemm.nativeIO,
   PublicIO.Gemm.SignedInputs,PublicIO.Gemm.initialValues,FP.Value,FP.Value.zero,FP.Format.fp32,
   PublicIO.Gemm.nativeFormula,IntegerDotPerAddressProgramIO.ImplementsResetOnCells]
#axiomsClean producer_native
#axiomsClean gemm_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.FPEquiv
