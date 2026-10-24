import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Boundary
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
open Classical

def logical (negative : Bool) (x : ℝ) : ℝ := if negative then -x else x

def tensorRead (v : TensorView) (s : BlockState) (x : Coordinate) : ℝ :=
  logical v.negative (s.readMem v.region (v.address x))

def doutRead (v : DoutView) (s : BlockState) (b c g k r : Nat) : ℝ :=
  logical v.negative (s.readMem v.region (v.address b c g k r))

/-- Independent initial logical Doutᵀ @ A plus optional residual. There is no
execution, scratch read, grid, or output state in this definition. -/
def expected (h : HostCall) (caller : Caller) (before : BlockState)
    (b c g r n : Nat) : ℝ :=
  (∑ k : Fin h.chunk,
    if k.val < min h.chunk (h.seqlen-c*h.chunk) then
      doutRead caller.dout before b c g k.val r *
        tensorRead caller.a before ⟨b,c*h.chunk+k.val,g,n⟩ else 0) +
  if h.hasResidual then tensorRead caller.residual before ⟨b,c*h.chunk+r,g,n⟩ else 0

def tensorExpected (h : HostCall) (caller : Caller) (before : BlockState) (x : Coordinate) : ℝ :=
  expected h caller before x.batch (x.sequence/h.chunk) x.group (x.sequence%h.chunk) x.feature

def coord (h : HostCall) (c : Tuning) (q : HostGrid.Pid) (i : TileIndex [c.BM,c.BN]) : Coordinate :=
  let p := HostContract.parameters h
  ⟨q.batch, RealViews.chunk p q.chunkHead*h.chunk+RealViews.row c.BM c.BN p q.tile i.1,
    RealViews.head p q.chunkHead, RealViews.column c.BN p q.tile i.2.1⟩

def targetAt (h : HostCall) (c : Tuning) (caller : Caller) (before : BlockState)
    (q : HostGrid.Pid) (i : TileIndex [c.BM,c.BN]) : ℝ :=
  let p := HostContract.parameters h
  expected h caller before q.batch (RealViews.chunk p q.chunkHead)
    (RealViews.head p q.chunkHead) (RealViews.row c.BM c.BN p q.tile i.1)
    (RealViews.column c.BN p q.tile i.2.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Boundary
