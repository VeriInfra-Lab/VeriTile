import bench.tritonbench_g.triton_argmax.Proofs.TypedWhole
import bench.tritonbench_g.triton_argmax.Proofs.Global

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def globalLoaded (s : State Word) (X : RegionName) (size block pid : Nat) (lane : Fin block) : Word :=
  if pid*block+lane.val<size then (s.mem X (pid*block+lane.val)).read .real else 0xff800000

def firstIndex (s : State Word) (X : RegionName) (size block base : Nat) (positive : 0<block) : Nat :=
  TypedReduction.selected positive (globalLoaded s X size block (s.pids 0+base))+(s.pids 0+base)*block

def firstWritten (s : State Word) (X V : RegionName) (I : Region .int)
    (size block base : Nat) (positive : 0<block) : State Word :=
  let selected := firstIndex s X size block base positive
  (s.write V (s.pids 0+base) (.mk .real ((s.mem X selected).read .real))).write
    (Region.cast I) (s.pids 0+base) (.mk .int (Int.ofNat selected))

def secondIndex (s : State Word) (V : RegionName) (size block : Nat) (positive : 0<block) : Nat :=
  TypedReduction.selected positive (fun lane : Fin block =>
    if lane.val<size then (s.mem V lane.val).read .real else 0xff800000)

def secondWritten (s : State Word) (V : RegionName) (I O : Region .int)
    (size block : Nat) (positive : 0<block) : State Word :=
  s.write (Region.cast O) 0 (.mk .int ((s.mem (Region.cast I) (secondIndex s V size block positive)).read .int))

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
