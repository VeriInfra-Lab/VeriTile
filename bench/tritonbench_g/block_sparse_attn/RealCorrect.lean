import bench.tritonbench_g.block_sparse_attn.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.Correct
open VeriTile Triton FP.Structural
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The denominator counts CSR occurrences, including repeated keys. Causal
masking is applied to each occurrence; every feature in the head participates. -/
def attentionWeight (s : BlockSparseAttn.Shape) (csr : BlockSparseAttn.CSR)
    (x : BlockSparseAttn.Inputs) (scale : ℝ) (b h row visit : Nat)
    (lane : Fin s.blockN) : ℝ :=
  let key := csr.key s h row visit lane.val
  if key ≤ (row : Int) then Real.exp
    (((List.range s.dim).map (fun d =>
      x.query b h row d * x.key b (BlockSparseAttn.kvHead s h) key d)).sum * scale)
  else 0

def mathematical (p : Parameters) (batches visits : Nat)
    (xs : PublicIO.LogicalInputs p batches visits) (cell : PublicIO.Coordinate p batches) : ℝ :=
  let s := xs.layout.shape
  let csr := xs.layout.csr
  let x := xs.tensors
  let w := cell.work
  (∑ visit ∈ Finset.range (csr.count s w.head w.row), ∑ lane : Fin s.blockN,
    attentionWeight s csr x p.scale w.batch w.head w.row visit lane *
      x.value w.batch (BlockSparseAttn.kvHead s w.head) (csr.key s w.head w.row visit lane.val) w.col) /
  (∑ visit ∈ Finset.range (csr.count s w.head w.row), ∑ lane : Fin s.blockN,
    attentionWeight s csr x p.scale w.batch w.head w.row visit lane)

/-- Full causal CSR occurrence quotient over every feature block and source
CTA. Repeated occurrences and masked tails remain independent input values;
metadata domain, strided read backing, source safety and frame are retained. -/
specification kernel_correctness (p : Parameters) (batches visits : Nat)
    (geometry : PublicIO.Geometry p) :
    Spec.Real (PublicIO.logicalIO p batches visits ⊨{PublicIO.logicalDomain p batches visits}
      mathematical p batches visits) :=
  PublicIO.logical_implements p batches visits geometry

#axiomsClean kernel_correctness
#specNonCircular mathematical avoiding [execution,output,accumulation,PhysicalKernel.grid]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.Correct
