import bench.repairs.tritonbench_g.log_softmax.IO
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native
import VeriTile.Triton.Float.ObservedProgramEquivalence

namespace VeriTile.Bench.Repairs.LogSoftmax.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems
noncomputable section

/-- All symbolic source operands are admitted, including leaves for every
FP32 bit pattern. The original load/cast/reduction graph stays intact. -/
def original (direction : Direction) (R : Native.Reductions) (M N K BM : Nat) :
    ProgramIO Native.Term Native.Term :=
  IO.grid direction id id (FP.ProfiledRewrite.engine (Native.algebra R))
    Bool.false M N K BM (Grid.blockColumns N)

def commuted (direction : Direction) (R : Native.Reductions) (M N K BM : Nat) :
    ProgramIO.Variant (original direction R M N K BM) :=
  IO.commuted direction id id (FP.ProfiledRewrite.engine (Native.algebra R))
    M N K BM (Grid.blockColumns N)

/-- A fixed native lowering/ISA interpretation observes the original graph.
No equation between fused and split rounding or exp/log implementations occurs. -/
def observation (direction : Direction) (R : Native.Reductions) (M N K BM : Nat)
    (I : Native.Instructions) (S : Native.Schedule) :
    (o : Fin (original direction R M N K BM).outputs.length) →
      TileIndex ((original direction R M N K BM).outputs[o]).shape →
      Native.Term → BitVec 32 := fun _ _ => Native.eval I S

end
end VeriTile.Bench.Repairs.LogSoftmax.NativeIO
