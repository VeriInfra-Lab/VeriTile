import bench.tritonbench_g.int_scaled_matmul.IO
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.IntScaledMatmul
open VeriTile.Triton FP.Structural Revision001
noncomputable section

namespace CandidateIntIO
def invocation (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) (pid : Nat) :
    Invocation KernelModel.Word :=
  { IntIO.invocation rest p pid with kernel := Kernels.unscaledCopyFree p }
def variant (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) :
    ProgramIO.Variant (IntIO.io rest p) where
  schedule := fun _ => (List.range (Geometry.count (BlockWord.geometry p))).map (invocation rest p)
  observe := fun _ _=>id
def io (rest : Algebra KernelModel.Word) (p : BlockWord.Parameters) := (variant rest p).toProgramIO
end CandidateIntIO

namespace CandidateScaledIO
def invocation (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) (pid : Nat) : Invocation KernelModel.Word :=
  { ScaledIO.invocation rest undefined p pid with kernel := Kernels.scaledSharedIndices p }
def variant (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) : ProgramIO.Variant (ScaledIO.io rest undefined p) where
  schedule := fun _ => (List.range (Geometry.count (ScaledWord.geometry p))).map (invocation rest undefined p)
  observe := fun _ _=>id
def io (rest : Algebra KernelModel.Word) (undefined : KernelModel.Word → BitVec 32)
    (p : ScaledWord.Parameters) := (variant rest undefined p).toProgramIO
end CandidateScaledIO

end
end VeriTile.Bench.Repairs.IntScaledMatmul
