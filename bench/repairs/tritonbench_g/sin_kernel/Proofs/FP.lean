import bench.repairs.tritonbench_g.sin_kernel.Proofs.Host
import VeriTile.Triton.Float.Structural
import bench.tritonbench_g.sin_kernel.Proofs.Source

namespace VeriTile.Bench.Repairs.SinKernel.FPProof
open VeriTile.Triton
open Binding (Word)
open VeriTile.Bench.TritonBenchG.AddExample.Finite (MemoryPrimitives)
noncomputable section

/-- Independent FP reference port with the source's explicit FP32 precision.
This is an opaque primitive interface, not an executable correctly-rounded sine. -/
def reference (target : FP.Structural.Algebra Word) (x : Word) : Word :=
  target.unary (some .fp32) .sin x

/-- Local primitive conformance, on all raw FP32 words including nonfinite values
and signed zero. It asserts no memory, index, coverage, or output predicate. -/
def LocalSine (nativeSin : Word → Word) (target : FP.Structural.Algebra Word) : Prop :=
  ∀ x, nativeSin x = reference target x

/-- Little-endian FP32 storage interpretation is an ordinary memory ABI law,
separate from sine. No Real-valued carrier is packed into four bytes. -/
def decode32 (bs : Nat → Fin 256) : Word :=
  FP.Value.ofBits .fp32 (bs 0 + 256*bs 1 + 65536*bs 2 + 16777216*bs 3)
def byte32 (x : Word) (k : Nat) : Fin 256 :=
  ⟨(x.bits.toNat / 2^(8*k)) % 256, Nat.mod_lt _ (by decide)⟩
structure FP32Storage {M : Type} (ops : MemoryPrimitives M Word) : Prop where
  decode : ∀ bs, ops.decode .fp32 bs = decode32 bs
  encode : ∀ value k, k < 4 → ops.encode .fp32 value k = byte32 value k

/-- The FP32 source erases to the original body interpreted by Source.runProgram. -/
theorem same_source (n : Nat) :
    (fp32Kernel n).eraseDType.toAlgKernel =
      (sourceKernel "x_ptr" "output_ptr" n 1024).toAlgKernel := by
  exact TritonBenchG.SinKernel.typed_projection .fp32 Bool.false _ _ _ _

/-- Source precision comes from the typed original, including absence of a cast. -/
theorem retained_precision (n : Nat) :
    (fp32Kernel n).surfaceBody.drop 4 =
      [TritonBenchG.SinKernel.Precision.expectedLoad .fp32 "x_ptr" 1024,
       TritonBenchG.SinKernel.Precision.expectedSin .fp32 1024,
       TritonBenchG.SinKernel.Precision.expectedStore .fp32 "output_ptr" 1024] :=
  TritonBenchG.SinKernel.Precision.source_precision .fp32 Bool.false _ _ _ _

#specNonCircular reference avoiding [Binding.runWrapper, Native.runJobs]
#specNonCircular LocalSine avoiding [Binding.runWrapper, Native.runJobs]
#axiomsClean same_source
#axiomsClean retained_precision
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.FPProof
