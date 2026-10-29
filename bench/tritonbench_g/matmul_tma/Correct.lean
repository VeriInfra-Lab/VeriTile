import bench.tritonbench_g.matmul_tma.Kernels
import bench.tritonbench_g.matmul_tma.Proofs.WholeReal

/-! Mathematical correctness of the actual FP16 source's real projection.
The wrapper has one program, BLOCK_M/N/K=M/N/K and all eight Boolean cases.
Floating numerical behavior, including WGMMA ordering, is separate. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.Correct
open VeriTile Triton FPValue Host
noncomputable section

/-- Public independent target, Σ_k A[i,k] B[k,j], for the logical tensor views. -/
abbrev matrixProduct := Real.product

/-- The real IO contract preserves the arbitrary disjoint flat placement of
all active buffers, initial inputs, output readback and every other flat cell.
The shared typed source projects to exactly this verified real program. -/
specification matmul_tma_real_correctness (h : Shape) (f : Flags) (r : Regions) :
    Spec.Real ((Real.io h f r).Implements (fun _ _ a b i => matrixProduct h a b i) ∧
      (source (parameters h f r) f.half).eraseDType.toAlgKernel =
        (Real.io h f r).kernel.toAlgKernel) :=
  ⟨Real.io_correct h f r, Real.source_real_projection (parameters h f r) f.half⟩

/-- The wrapper's actual contiguous/transposed allocation sizes discharge all
unmasked block-pointer bounds; output injectivity is derived, not assumed. -/
specification matmul_tma_wrapper_layout (h : Shape) (f : Flags) (r : Regions)
    (a : FlatAlloc) (binding : AllocationBinding h f r a) :
    (∀ i, aOffset (parameters h f r) i < a.extent (parameters h f r).a) ∧
    (∀ i, bOffset (parameters h f r) i < a.extent (parameters h f r).b) ∧
    (∀ i, FPValue.cOffset (parameters h f r) i < a.extent (parameters h f r).c) ∧
    Function.Injective (FPValue.cOffset (parameters h f r)) := by
  obtain ⟨ha,hb,hc⟩ := all_accesses_bounded h f r a binding
  exact ⟨ha,hb,hc,c_injective h f r⟩

/-- Every cell of the returned M×N tensor holds the independent mathematical
product in the real projection; all non-output cells are preserved. -/
specification matmul_tma_real_return (h : Shape) (f : Flags) (r : Regions)
    (s : BlockState) (a : TileIndex [h.rows,h.inner] → ℝ)
    (b : TileIndex [h.inner,h.columns] → ℝ)
    (ha : ∀ i, s.readMem (parameters h f r).a (aOffset (parameters h f r) i) = a i)
    (hb : ∀ i, s.readMem (parameters h f r).b (bOffset (parameters h f r) i) = b i) :
    Spec.Real (∃ t,
      exec (source (parameters h f r) f.half).eraseDType.toAlgKernel s = some t ∧
      (∀ i, t.readMem (returnedTensor h f r).region
        (i.1.val * h.columns + i.2.1.val) = matrixProduct h a b i) ∧
      (∀ R off, R ≠ (returnedTensor h f r).region ∨
        (∀ i : TileIndex [h.rows,h.columns], off ≠ i.1.val * h.columns + i.2.1.val) →
        t.mem R off = s.mem R off)) := by
  simpa only [returned_region] using Real.execution h f r s a b ha hb

/-- Actual full allocation placement, including the unused first FP32 buffer
on the FP16 branch, with all byte-independent flat cells framed. -/
specification matmul_tma_real_flat_return (h : Shape) (f : Flags) (r : Regions)
    (s : BlockState) (a : TileIndex [h.rows,h.inner] → ℝ)
    (b : TileIndex [h.inner,h.columns] → ℝ)
    (ha : ∀ i, s.readMem (parameters h f r).a (aOffset (parameters h f r) i) = a i)
    (hb : ∀ i, s.readMem (parameters h f r).b (bOffset (parameters h f r) i) = b i)
    (storage : FlatAlloc) (binding : AllocationBinding h f r storage)
    (coverage : ∀ R, R ∉ storage.regions → storage.extent R = 0)
    (clean : s.undef = fun _ _ => 0) :
    Spec.Real (∃ t,
      exec (storage.flattenKernel (source (parameters h f r) f.half).eraseDType.toAlgKernel)
        (storage.flattenState s) = some t ∧
      (∀ i, t.readMem storage.flat (storage.addr (outputRegion r f)
        (i.1.val * h.columns + i.2.1.val)) = matrixProduct h a b i) ∧
      (∀ R off, R ≠ storage.flat ∨
        (∀ i : TileIndex [h.rows,h.columns],
          off ≠ storage.addr (outputRegion r f) (i.1.val * h.columns + i.2.1.val)) →
        t.mem R off = (storage.flattenState s).mem R off)) :=
  Real.flat_execution h f r s a b ha hb storage binding coverage clean

/-- The mathematical interpretation of the complete original wrapper:
actual allocation order, admitted source launch, selected returned tensor,
independent matrix product, discarded-buffer frame and pre-allocation frame.
Storage conversions are erased here; native numerical claims stay separate. -/
specification matmul_tma_whole_wrapper_real
    (rest : FP.Structural.Algebra ℝ) (alloc : Allocator ℝ) (admit : Admission)
    (h : Shape) (f : Flags) (cfg : Config) (r : Regions) (storage : FlatAlloc)
    (initial s : FP.Structural.State ℝ) (prepared : Prepared alloc h f r initial s)
    (binding : AllocationBinding h f r storage) (frame : AllocatorFrame alloc)
    (ready : admit (request h f r cfg) = .ok ()) :
    Spec.Real (∃ t,
      wrapper (WholeReal.algebra rest) WholeReal.dot alloc admit h f cfg initial =
        .ok (returnedTensor h f r,t) ∧
      Post (parameters h f r) f.half s t
        (matrixProduct h (aValues (parameters h f r) s) (bValues (parameters h f r) s)) ∧
      (f.half = Bool.true → ∀ off, t.mem r.first off = s.mem r.first off) ∧
      (∀ R off, OutsideAllocated f r R → t.mem R off = initial.mem R off)) :=
  WholeReal.wrapper_mathematical rest alloc admit h f cfg r storage initial s prepared binding frame ready

#axiomsClean matmul_tma_whole_wrapper_real
#axiomsClean matmul_tma_real_flat_return
#axiomsClean matmul_tma_real_correctness
#axiomsClean matmul_tma_wrapper_layout
#axiomsClean matmul_tma_real_return
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Correct
