import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.IO

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

/-- The independent full-D causal CSR occurrence-multiset quotient. The
finite arrays merely provide the original strided logical input views. -/
def realFormula (p : Parameters) (batches visits : Nat) (xs : Inputs ℝ p batches visits) :
    Outputs ℝ p batches := fun
  | ⟨0,_⟩ => fun j =>
    let x := targetLayout p batches visits xs
    let w := coordinate p batches j
    BlockSparseAttn.Spec.output x.shape x.csr (BlockSparseAttn.Real.inputs x (memory 0 p batches visits xs))
      p.scale w.batch w.head w.row w.col

/-- Original ordered native scalar recurrence, fixed K8 instructions, physical
reduction replicas and raw scale word. No Real equality or new rewrite law. -/
def nativeFormula (ops : PhysicalKernel.NativePrimitives) (p : Parameters) (batches visits : Nat)
    (xs : Inputs Word p batches visits) : Outputs Word p batches := fun
  | ⟨0,_⟩ => fun j =>
    rowValue (targetLayout p batches visits xs) (BlockSparseAttn.Native.ops ops.instructions)
      (memory (0 : Word) p batches visits xs) ops.scale (coordinate p batches j)

#specNonCircular realFormula avoiding [execution,output,accumulation,PhysicalKernel.grid]
#specNonCircular nativeFormula avoiding [execution,output,accumulation,PhysicalKernel.grid]
#specNonCircular Admissible avoiding [execution,output,accumulation,PhysicalKernel.grid,realFormula,nativeFormula]
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PublicIO
