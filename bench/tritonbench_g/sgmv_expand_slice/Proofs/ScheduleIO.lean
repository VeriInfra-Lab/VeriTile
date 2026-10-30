import bench.tritonbench_g.sgmv_expand_slice.Sources.Kernels
import bench.tritonbench_g.sgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open VeriTile.Triton

structure Job where
  (batch pid count : Nat)
  deriving DecidableEq, Repr

def batchJobs (columns batch : Nat) (segment : Segment) : List Job :=
  if segment.active ∧ 0 < columns then
    let count := tiles segment.length * tiles columns
    (List.range count).map (fun pid => ⟨batch,pid,count⟩)
  else []

def jobsFrom (columns : Nat) : Nat → List Segment → List Job
  | _,[] => []
  | batch,s::ss => batchJobs columns batch s ++ jobsFrom columns (batch+1) ss

def jobs (columns : Nat) (segments : List Segment) := jobsFrom columns 0 segments

end VeriTile.Bench.Repairs.SgmvExpandSlice
