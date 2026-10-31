import VeriTile.Triton.DSL.CallSignature
import bench.tritonbench_g.matmul_triton1.Proofs.SourceParts
import bench.tritonbench_g.matmul_triton1.Sources.DotProfile
import VeriTile.Meta.StatementAudit

/-! Pasted Python tuple shapes and zero-start ranges preserve the complete
program, including nested loops, stores, runtime casts and dot options. -/
namespace VeriTile.Tests.RawBodySurface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulTriton1
open Revision001

def signature : CallSignature where
  pointers := [("out",.fp32)]
  constexpr := ["M","N"]

def tupleForm (out : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    z = tl.zeros((M, N), dtype=tl.float32)
    for i in range(0, M) {
      for j in range(0, N, 1) {
        z += tl.full([M, N], 0.0, dtype=tl.float32)
      }
    }
    ptrs = out + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(ptrs, z)
  }

def listForm (out : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    z = tl.zeros([M, N], dtype=tl.float32)
    for i in range($(0), M, $(1)) {
      for j in range($(0), N, 1) {
        z += tl.full([M, N], 0.0, dtype=tl.float32)
      }
    }
    ptrs = out + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(ptrs, z)
  }

theorem whole_program (out : RegionName) (M N : Nat) :
    tupleForm out M N=listForm out M N := by rfl

/-- The runtime ABI cast remains an actual modulo operation, even outside
correctness's public non-overflow domain. -/
theorem scalar_wrap {α : Type} [Inhabited α] (rest : Algebra α) (P : Core.Primitives α)
    (p : Core.Parameters) (s : State (DotProfile.Value α)) :
    (SourceRuntime.preludeExecution rest P {p with rows := 2^64+5} s).val.regs
      .nat [] "m_size"=some (fun _ => 5) := by
  simpa using SourceRuntime.prelude_rows rest P {p with rows := 2^64+5} s

/-- A backend bound to allow_tf32=False cannot silently interpret another
precision profile as the same dot. -/
theorem reject_other_dot_options {α : Type} (rest : Algebra α) (P : Core.Primitives α)
    {m k n : Nat} (xs : TileIndex [m,k] → DotProfile.Value α)
    (ys : TileIndex [k,n] → DotProfile.Value α) :
    (DotProfile.algebra rest P).dotWithOptions (some .fp32) {allowTF32 := some Bool.true} (batch:=[]) (m:=m) (k:=k) (n:=n) xs ys=none := by
  simp [DotProfile.algebra]

#axiomsClean whole_program
#axiomsClean scalar_wrap
#axiomsClean reject_other_dot_options
end VeriTile.Tests.RawBodySurface
