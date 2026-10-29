import bench.repairs.tritonbench_g.int4_matmul.Kernels
import bench.repairs.tritonbench_g.int4_matmul.Contract
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Kernel.TensorResults
import bench.repairs.tritonbench_g.int4_matmul.Misc.HistoricalCorrect
import bench.repairs.tritonbench_g.int4_matmul.Misc.HistoricalFPEquiv
import bench.repairs.tritonbench_g.int4_matmul.Misc.RawCorrect
import bench.repairs.tritonbench_g.int4_matmul.Misc.RawNative

/-! Complete retained checked Tensor applications. -/
namespace VeriTile.Bench.Repairs.Int4Matmul.TensorBridge
open VeriTile.Triton Direct Host KernelExecution
noncomputable section

/-- The canonical FP32 kernel computes the independent dense INT4 formula at
every original Tensor coordinate, with complete input/exterior storage frame.
The source wrapper supplies the checked descriptors and zeroed output boundary;
allocation, return identity and runtime internals are not proof obligations here. -/
theorem matmul_kernel_correct (r : MatmulRequest) (out : TensorInfo)
    (output : OutputBinding r out) (tile : Tile) (member : tile∈sourceTiles)
    (rest : FP.Structural.Algebra ℝ) (initial : HeapState ℝ)
    (scratch : RegisterState ℝ) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes r.shape tile)) :
    Spec.Real (
      (∀ row col,row<r.shape.m → col<r.shape.n →
        (realKernel r out tile rest initial scratch lanes).values out.descriptor.allocation (logicalIndex out row col)=
          realTarget r out initial row col) ∧
      (∀ storage i,storage ≠ out.descriptor.allocation ∨ r.shape.m * r.shape.n ≤ i →
        (realKernel r out tile rest initial scratch lanes).values storage i=initial.values storage i) ∧
      (∀ input∈r.views,∀ i,(realKernel r out tile rest initial scratch lanes).values input.descriptor.allocation i=
        initial.values input.descriptor.allocation i) ∧
      (realKernel r out tile rest initial scratch lanes).packed=initial.packed) :=
  real_tensor_result r out output tile member rest initial scratch lanes complete

#axiomsClean matmul_kernel_correct
#specNonCircular realTarget avoiding [KernelSource.kernel,launch,parallel,computed,realKernel]
end
end VeriTile.Bench.Repairs.Int4Matmul.TensorBridge

namespace VeriTile.Bench.Repairs.Int4Matmul.TensorBridge
open VeriTile.Triton Direct Host Physical KernelExecution
noncomputable section

/-- Exact native instruction interpretation of the same quoted FP32 source.
Only bounded scalar/fixed-K8 primitive semantics and the selected ISA are model
parameters. All source loads, live carries, masks, split atomics, actual Tensor
addresses, output bytes and exterior frame are proved. No finite-input or
whole-dot/result premise occurs, and no primitive obligation is an experiment. -/
theorem matmul_kernel_native (base : Nat → Nat) (r : MatmulRequest) (out : TensorInfo)
    (output : OutputBinding r out) (inputs : ∀ input∈r.views,TensorLaws base input)
    (outLaws : TensorLaws base out) (tile : Tile) (member : tile∈sourceTiles)
    (instructions : Native.Instructions) (isa : Native.ISA) (rest : FP.Structural.Algebra Word)
    (initial : ByteHeap) (scratch : RegisterState Word) (lanes : List Lane)
    (complete : lanes.Perm (rawLanes r.shape tile)) :
    (∀ input∈r.views,PointerBinding base input) ∧ PointerBinding base out ∧
    (∀ row col,row<r.shape.m → col<r.shape.n → ∀ byte : Fin 4,
      nativeKernel r out tile instructions isa rest initial scratch lanes out.descriptor.allocation
        (4*logicalIndex out row col+byte.val)=
      encode (nativeTarget r out tile instructions isa initial lanes row col) byte) ∧
    (∀ storage b,storage ≠ out.descriptor.allocation ∨ r.shape.m * r.shape.n ≤ b/4 →
      nativeKernel r out tile instructions isa rest initial scratch lanes storage b=initial storage b) :=
  native_tensor_result base r out output inputs outLaws tile member instructions isa rest initial scratch lanes complete

#axiomsClean matmul_kernel_native
#specNonCircular nativeTarget avoiding [KernelSource.kernel,launch,parallel,computed,nativeKernel,partialValue]
end
end VeriTile.Bench.Repairs.Int4Matmul.TensorBridge

namespace VeriTile.Bench.Repairs.Int4Matmul.TensorBridge
open VeriTile.Triton Direct Host KernelExecution
noncomputable section

/-- Exact successful Python guard domain, reused without adding a finite-A,
finite-scale, divisible-tile-K or selected-specialization restriction. -/
theorem python_checks_iff_preconditions (input weight scales zeros : TensorInfo)
    (group : Int) (out : Option TensorInfo) :
    checkMatmul input weight scales zeros group out=.ok () ↔ MatmulContract input weight scales zeros group out :=
  checkMatmul_iff input weight scales zeros group out

#axiomsClean python_checks_iff_preconditions
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int4Matmul.TensorBridge
