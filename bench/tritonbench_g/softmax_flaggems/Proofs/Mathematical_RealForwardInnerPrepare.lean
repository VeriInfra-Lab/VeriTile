import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerReduce
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource

structure ReducedInv (sb : BlockState) (X Y : RegionName)
    (row N B : Nat) (hB : 0 < B) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  inputReg : st.regs .ptr [] "input_ptr" = some (Tile.scalar (X, row * N))
  outputReg : st.regs .ptr [] "output_ptr" = some (Tile.scalar (Y, row * N))
  previousReg : st.regs .nat [] "previous_multiple" = some (Tile.scalar (RealAddress.previous N B))
  maxReg : st.regs .real [] "m" = some (Tile.scalar (finalMax sb X row N B hB))
  sumReg : st.regs .real [] "z" = some (Tile.scalar (some (finalSum sb X row N B hB)))

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
