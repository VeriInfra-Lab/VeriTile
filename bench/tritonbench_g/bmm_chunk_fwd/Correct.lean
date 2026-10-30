import bench.tritonbench_g.bmm_chunk_fwd.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.BmmChunkFwd.Correct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

def fullProduct (h : Host.Call) (x : Host.Inputs h) (batch chunk head row col : Nat) : ℝ :=
  if row < min h.shape.chunk (h.shape.seqlen - chunk * h.shape.chunk) ∧
      col < min h.shape.chunk (h.shape.seqlen - chunk * h.shape.chunk) ∧
      (h.hasSeq = Bool.false ∨ x.seq batch (chunk * h.shape.chunk + row) =
        x.seq batch (chunk * h.shape.chunk + col))
  then ∑ k : Fin h.shape.K,
    x.a batch (chunk * h.shape.chunk + row) head k *
    x.b batch (chunk * h.shape.chunk + col) head k
  else 0

def formula (p : Parameters) (xs : BmmIO.LogicalInputs p) (cell : BmmIO.Coordinate p) : ℝ :=
  if p.causal=Bool.true ∧ cell.2.2.2.1.val < cell.2.2.2.2.val then 0 else
    fullProduct (parameters p) xs.tensors cell.1.val cell.2.1.val cell.2.2.1.val
      cell.2.2.2.1.val cell.2.2.2.2.val

/-- Complete product with sequence filtering, padding and causal zeros at
batch/chunk/group/row/column coordinates. Full input envelopes retain unused
gaps and arbitrary sequence strides; main grid, cleanup, safety and frame remain. -/
specification bmm_chunk_fwd_real (p : Parameters) (c : Host.Config) (hc : c ∈ Host.configurations) :
    Spec.Real (BmmIO.logicalIO p c ⊨ formula p) :=
  BmmIO.logical_implements p c hc

#axiomsClean bmm_chunk_fwd_real
#specNonCircular formula avoiding [Kernels.fp32Source,Kernels.source,KernelReal.Raw.Runs,exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.Correct
