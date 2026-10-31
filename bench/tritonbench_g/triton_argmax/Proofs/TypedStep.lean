import bench.tritonbench_g.triton_argmax.Proofs.TypedBackend
import bench.tritonbench_g.triton_argmax.Proofs.Dimension

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def loaded (s : State Word) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (i : Fin BM) (j : Fin BN) : Word :=
  if row+i.val<M ∧ cur+j.val<N then (s.mem X ((row+i.val)*N*K+(cur+j.val)*K+inner)).read .real
  else 0xff800000

def next (s : State Word) (X : RegionName) (M N K BM BN row inner cur : Nat)
    (positive : 0<BN) (current : Fin BM → Word × Nat) (i : Fin BM) : Word × Nat :=
  let column := cur+TypedReduction.selected positive (loaded s X M N K BM BN row inner cur i)
  let value : Word := if row+i.val<M ∧ column<N then
    (s.mem X ((row+i.val)*N*K+column*K+inner)).read .real else 0xff800000
  if greater value (current i).1 then (value,column) else current i

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
