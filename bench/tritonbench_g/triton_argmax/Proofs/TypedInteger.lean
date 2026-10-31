import bench.tritonbench_g.triton_argmax.Proofs.IntegerReduction
import bench.tritonbench_g.triton_argmax.Proofs.TypedGlobal

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedInteger
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def selected (s : State Word) (V : Region .int) (size block : Nat) (padding : Int) : Nat :=
  TileShape.leftmostIntArgMax (fun lane : Fin block =>
    if lane.val<size then (s.mem (Region.cast V) lane.val).read .int else padding)

theorem selected_tree (s : State Word) (V : Region .int) (size block : Nat) (padding : Int)
    (positive : 0<block) (tree : Tree block) (complete : tree.Complete) :
    selected s V size block padding=(tree.pick orderedGt orderedEq
      (fun lane => if lane.val<size then (s.mem (Region.cast V) lane.val).read .int else padding)).val :=
  IntegerReduction.shared_tree positive _ tree complete

#axiomsClean selected_tree
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedInteger
