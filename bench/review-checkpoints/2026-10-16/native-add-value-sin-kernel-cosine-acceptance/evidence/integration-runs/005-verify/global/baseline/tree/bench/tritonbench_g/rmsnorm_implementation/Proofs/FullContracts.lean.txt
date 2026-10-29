import bench.tritonbench_g.rmsnorm_implementation.Proofs.Contracts
import bench.tritonbench_g.rmsnorm_implementation.Proofs.Safety

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton

/-- Actual trace safety for all initial states with the declared active access
bounds. The predicate does not assume successful execution or output values. -/
def TraceSafeIO (io : StreamEmitMasked2DKernelIO₂) : Prop :=
  ∀ (bounds : RegionBounds) (s : BlockState),
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
      io.read1 (s.pids 0) (s.pids 1) t j < bounds io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
      io.read2 (s.pids 0) (s.pids 1) t j < bounds io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask (s.pids 0) (s.pids 1) t j →
      io.write (s.pids 0) (s.pids 1) t j < bounds io.out) →
    io.kernel.toAlgKernel.TraceSafeR .triv bounds s

/-- Complete erased-real IO: actual execution-trace safety, successful finite
flat execution, the independent full-row formula and a frame for every raw cell. -/
def FullExactIO (io : StreamEmitMasked2DKernelIO₂)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.T → Fin io.C → ℝ) : Prop :=
  TraceSafeIO io ∧ ExactIO io f

/-- Both complete source loops satisfy the named safety and flat-memory
contract; shape, strides, all three floating storage labels and epsilon remain
symbolic. The real projection erases storage rounding and fixes undefined
masked load values to zero only in its execution component. -/
theorem typed_full_exact_io (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXO : X ≠ O) (hWO : W ≠ O) (hsok : 0 < sok) :
    FullExactIO (typedIO input weight output v X W O sxb sxm sxk sw sob som sok N B eps)
      (fun _ _ xs ws t j => rmsImplStreamSpec N B eps xs ws t j) := by
  refine ⟨?_, typed_exact_io input weight output v X W O sxb sxm sxk sw sob som sok N B eps
    hB hXO hWO hsok⟩
  intro bounds s hx hw ho
  exact typed_trace_safe input weight output v bounds X W O sxb sxm sxk sw sob som sok N B eps
    hB hXO hWO hsok s hx hw ho

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
