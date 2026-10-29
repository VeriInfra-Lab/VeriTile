import bench.repairs.tritonbench_g.int_scaled_matmul.Proofs.CandidateIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.IntScaledMatmul.FPEquiv
open VeriTile.Triton FP.Structural Revision001
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited KernelModel.Word := ⟨FP.Value.zero .fp32⟩
def unscaledOriginal (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) :=
  ProgramIO.Implementation.original (IntIO.io rest p) (fun _=>True)
def unscaledCopyFree (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) :=
  ProgramIO.Implementation.ofVariant (CandidateIntIO.variant rest p) (fun _=>True)
def scaledOriginal (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) :=
  ProgramIO.Implementation.original (ScaledIO.io rest undefined p) (fun _=>True)
def scaledSharedIndices (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) :=
  ProgramIO.Implementation.ofVariant (CandidateScaledIO.variant rest undefined p) (fun _=>True)

/-- Eliminate the copied accumulator, preserving the full grouped grid and every
signed32 output, source access safety and both complete exterior frames. -/
specification int_matmul_pair (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters)
    (geometry : Geometry.Valid (BlockWord.geometry p)) (positive : 0<p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    unscaledOriginal rest p ≡[[]] unscaledCopyFree rest p := by
  apply ProgramIO.Implementation.of_equivalent
  exact ProgramIO.EquivalentOn.of_implementsOn (IntIO.complete rest p geometry positive ha hb)
    (CandidateIntIO.complete rest p geometry positive ha hb)

/-- Reuse the prelude's unchanged row/column indices. The signed8 dot, modular
carry, FP32 promotion/multiplication/conversion, masks and all scale words stay fixed. -/
specification int_scaled_matmul_pair (rest : Algebra KernelModel.Word)
    (undefined : KernelModel.Word → BitVec 32) (p : ScaledWord.Parameters)
    (geometry : Geometry.Valid (ScaledWord.geometry p))
    (even : p.even=Bool.true → p.inner=p.steps*p.blockInner) (cover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) :
    scaledOriginal rest undefined p ≡[[]] scaledSharedIndices rest undefined p := by
  apply ProgramIO.Implementation.of_equivalent
  exact ProgramIO.EquivalentOn.of_implementsOn (ScaledIO.complete rest undefined p geometry even cover ha hb hs)
    (CandidateScaledIO.complete rest undefined p geometry even cover ha hb hs)

#print_fp_assumptions int_matmul_pair
#print_fp_assumptions int_scaled_matmul_pair
#axiomsClean int_matmul_pair
#axiomsClean int_scaled_matmul_pair
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IntScaledMatmul.FPEquiv
