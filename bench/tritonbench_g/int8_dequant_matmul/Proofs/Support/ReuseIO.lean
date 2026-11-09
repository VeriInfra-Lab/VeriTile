import bench.tritonbench_g.int8_dequant_matmul.Proofs.ReuseIndices
import bench.tritonbench_g.int8_dequant_matmul.Proofs.IntegerPairIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.ReuseIO
open VeriTile Triton FP.Structural
noncomputable section

def invocation (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (p : IOPrograms.Program s c) : IntegerDotInvocation (Sum F H) :=
  { IO.invocation q s c hasBias divisor p with kernel := ReuseIndices.source s c hasBias }

def variant (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (schedule : Schedule s c) :
    PairIO.Variant (IO.programIO q s c hasBias divisor schedule) where
  invocation := fun _=>invocation q s c hasBias divisor
  observe := fun _ _=>Quote.asH q.ops

def original (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (schedule : Schedule s c) :=
  PairIO.Implementation.original (IO.programIO q s c hasBias divisor schedule) (IOLayout.domain q.ops s c)

def optimized (q : Quote.IntegerModel I F H) (s : Shape) (c : Config)
    (hasBias : Bool) (divisor : F) (schedule : Schedule s c) :=
  PairIO.Implementation.ofVariant (variant q s c hasBias divisor schedule) (IOLayout.domain q.ops s c)

end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.ReuseIO
