import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Kernels
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawSource
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawAutotune

/-! Candidate: chunked_cumsum_fwd_repaired.py. Original source SHA256:
a3001271cff5c60be7e30cca02aac605685237016dbcdd1b3e14ac686b72efac.
The complete floating source is reused verbatim: optional bias, eager cutoff
softplus, ordered maximum/minimum with runtime words, remask, dt store, A load,
multiply, inclusive row scan, cumulative store. Only host input materialization,
exact integer ceiling, and native integer representation change. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd

abbrev source := RawSource.source
noncomputable abbrev run := @RawBoundary.tunedRun
abbrev Host := Whole.Host
abbrev Launch := Math.Launch
abbrev Word := BitVec 32

theorem template_eq (c : Whole.Host) (r : Program.Regions)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    WholeFP.typed32 c r Bool.false hasBias useSoftplus lower upper=
      fp32Source r.dt r.rate r.bias r.out r.sum c.batches c.sequence c.heads c.chunk lower upper
        c.layout.dtBatch c.layout.dtTime c.layout.dtHead c.layout.rateHead c.layout.biasHead
        c.layout.outBatch c.layout.outChunk c.layout.outHead c.layout.outColumn
        c.layout.sumBatch c.layout.sumChunk c.layout.sumHead c.layout.sumColumn
        useSoftplus hasBias c.BH c.BC := rfl

namespace Index
abbrev u (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
/-- The actual widened source expressions, with pointer-add grouping retained. -/
def head (c : Host) (l : Launch) (i : Nat) := u l.head * u c.BH + u i
def remaining (c : Host) (l : Launch) := u c.sequence - u l.chunk * u c.chunk
def dt (c : Host) (l : Launch) (i j : Nat) :=
  u l.batch*u c.dtBatch + u l.chunk*u c.chunk*u c.dtTime +
    (head c l i*u c.dtHead + u j*u c.dtTime)
def rate (c : Host) (l : Launch) (i : Nat) := head c l i*u c.rateHead
def bias (c : Host) (l : Launch) (i : Nat) := head c l i*u c.biasHead
def out (c : Host) (l : Launch) (i j : Nat) :=
  u l.batch*u c.layout.outBatch + u l.chunk*u c.layout.outChunk +
    (head c l i*u c.layout.outHead + u j*u c.layout.outColumn)
def address (dataPtr : Nat) (off : BitVec 64) := u dataPtr + off*u 4
end Index
end VeriTile.Bench.Repairs.ChunkedCumsumFwd
