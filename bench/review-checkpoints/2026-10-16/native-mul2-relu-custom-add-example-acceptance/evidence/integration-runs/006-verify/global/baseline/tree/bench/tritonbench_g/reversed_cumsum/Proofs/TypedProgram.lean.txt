import bench.tritonbench_g.reversed_cumsum.Proofs.Source
import bench.tritonbench_g.reversed_cumsum.Proofs.RealProgram
import bench.tritonbench_g.reversed_cumsum.Proofs.TypedModel

/-! The actual typed source split at its signed loop. All casts and every
statement are retained. These helpers only expose the original source tree. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedProgram
open VeriTile.Triton RealProgram
set_option maxHeartbeats 1500000
set_option maxRecDepth 16000

def parts (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) : Source.IntRange :=
  (Source.intRange? (TypedSource.source swapped dtype out_dtype s o
    p.headStride p.timeStride p.channelStride p.T p.S p.BT p.BS).surfaceBody
      "__rev_i_t").getD ⟨[], .constInt 0, .constInt 0, .constNat 0, [], []⟩

def startOp (p : Layout) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.castNatToInt (stopOp p)) (.constInt 1))

theorem source_split (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    (TypedSource.source swapped dtype out_dtype s o p.headStride p.timeStride
      p.channelStride p.T p.S p.BT p.BS).surfaceBody =
    (parts swapped dtype out_dtype p s o).before ++
      [.forRangeInt "__rev_i_t" (startOp p) (.constInt 1) (.constNat 1)
        (parts swapped dtype out_dtype p s o).body] := by
  cases swapped <;> rfl

theorem common_loop (dtype out_dtype : FloatDType) (p : Layout) (s o : RegionName) :
    (parts Bool.false dtype out_dtype p s o).body =
      (parts Bool.true dtype out_dtype p s o).body := rfl

theorem body_length (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) :
    (parts swapped dtype out_dtype p s o).body.length = 7 := by
  cases swapped <;> rfl

#axiomsClean source_split
#axiomsClean common_loop
#axiomsClean body_length
end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedProgram
