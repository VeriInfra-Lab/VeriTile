import bench.repairs.tritonbench_g.int_scaled_matmul.Correct
import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.NativeComplete
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Correct
open VeriTile.Triton FP.Structural Revision001
noncomputable section
attribute [local instance] wordDefault

/-- The second existing kernel adds FP32 row scaling and total signed32 output
conversion. Empty shapes, K tails, overflow and all scale words stay admitted. -/
theorem int_scaled_matmul_correct (rest : Algebra KernelModel.Word)
    (undefined : KernelModel.Word → BitVec 32) (p : ScaledWord.Parameters)
    (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) :
    (ScaledIO.io rest undefined p).Implements (ScaledIO.formula undefined p) :=
  ScaledIO.complete rest undefined p geometry even cover ha hb hs

#specNonCircular ScaledIO.formula avoiding [Kernels.scaled,TotalScaled.logicalResult,FP.Structural.exec,runSchedule]
#stmtSurfaceSubset int_scaled_matmul_correct ⊆
  [RegionName,KernelModel.Word,FP.Value,FP.Structural.Algebra,ScaledWord.Parameters,Geometry.Valid,ScaledWord.geometry,
    ScaledWord.Parameters.even,ScaledWord.Parameters.inner,ScaledWord.Parameters.steps,
    ScaledWord.Parameters.blockInner,ScaledWord.Parameters.input,ScaledWord.Parameters.weight,
    ScaledWord.Parameters.output,ScaledWord.Parameters.scale,ScaledIO.io,ScaledIO.formula,
    ProgramIO.Implements,wordDefault]
#axiomsClean int_scaled_matmul_correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Correct

namespace VeriTile.Bench.Repairs.IntScaledMatmul.FPEquiv
open VeriTile.Triton FP.Structural Revision001 FiniteExecution
noncomputable section
local instance wordDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete source/native equivalence with fixed K32 saturating instructions
and separately wrapping signed32 carry. The same variant is transported to the
original finite signed-index executor by NativeComplete.int_transport. -/
theorem int_matmul_equiv (rest : Algebra Word) (x : Launch)
    (legal : Contract.ConfigContract x.config)
    (envelope : x.M=0 ∨ x.N=0 ∨ Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (BlockWord.geometry (blockParameters x)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) :
    (IntIO.io rest (blockParameters x)).Equivalent (NativeIO.intVariant rest x) :=
  NativeComplete.int_equivalent rest x legal geometry ha hb

/-- The scaled source additionally retains exact FP32 promotion, the inspected
reversed multiply operands, and the original total cast port. The all-word
multiply identity is proved, with no deferred numerical assumption. The same
variant reaches CheckedNative through NativeComplete.scaled_transport. -/
theorem int_scaled_matmul_equiv (rest : Algebra Word) (undefined : Word → BitVec 32)
    (scale : RegionName) (x : Launch) (legal : Contract.ConfigContract x.config)
    (envelope : x.M=0 ∨ x.N=0 ∨ Contract.IndexEnvelope x.M x.N x.K x.config)
    (geometry : Geometry.Valid (ScaledWord.geometry (scaledParameters x scale)))
    (ha : x.a≠x.c) (hb : x.b≠x.c) (hs : scale≠x.c) :
    (ScaledIO.io rest undefined (scaledParameters x scale)).Equivalent
      (NativeIO.scaledVariant rest undefined x scale) :=
  NativeComplete.scaled_equivalent rest undefined x scale legal geometry ha hb hs

#stmtSurfaceSubset int_matmul_equiv ⊆
  [RegionName,Word,KernelModel.Word,FP.Value,FP.Structural.Algebra,Launch,Launch.config,Launch.M,Launch.N,Launch.K,
    Launch.a,Launch.b,Launch.c,Contract.ConfigContract,Contract.IndexEnvelope,
    Geometry.Valid,BlockWord.geometry,blockParameters,IntIO.io,NativeIO.intVariant,
    ProgramIO.Equivalent,wordDefault]
#stmtSurfaceSubset int_scaled_matmul_equiv ⊆
  [RegionName,Word,KernelModel.Word,FP.Value,FP.Structural.Algebra,Launch,Launch.config,Launch.M,Launch.N,Launch.K,
    Launch.a,Launch.b,Launch.c,Contract.ConfigContract,Contract.IndexEnvelope,
    Geometry.Valid,ScaledWord.geometry,scaledParameters,ScaledIO.io,NativeIO.scaledVariant,
    ProgramIO.Equivalent,wordDefault]
#axiomsClean int_matmul_equiv
#axiomsClean int_scaled_matmul_equiv
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IntScaledMatmul.FPEquiv
