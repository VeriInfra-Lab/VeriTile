import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PositionalReductionLoop
open VeriTile.Triton FP.Structural
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def signature : CallSignature where
  pointers := [("X",.fp32),("Copy",.fp32),("Y",.fp32)]
  typedFP32Storage := Bool.true
  constexpr := ["N","B"]

def source (X Copy Y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    peak = float("-inf")
    total = 0.0
    lanes = tl.arange(0, B)
    for i in range(N):
      x = tl.load(X + i * B + lanes)
      tl.store(Copy + i * B + lanes, x)
      next_peak = tl.maximum(tl.max(x, 0), peak)
      weight = tl.exp(x - next_peak)
      total = total + tl.sum(weight, 0)
      peak = next_peak
    tl.store(Y, total)
  }

/-- The memory representation remains FP32, including the direct copy. Only
reduction/arithmetic uses the FP32-to-real representation projection. No
numerical identity about a cast or the reduction callback is assumed. -/
def expanded (X Copy Y : RegionName) (N B : Nat) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  triton {
    tl.ports inputs=[X], outputs=[Copy,Y]
    tl.region X = DTYPE, Copy = DTYPE, Y = DTYPE
    peak = float("-inf")
    total = tl.full([], 0.0, dtype=tl.float32)
    lanes = tl.arange(0, $(B))
    tl.for i in $(N) {
      x = tl.load(X + i * $(B) + lanes)
      tl.store(Copy + i * $(B) + lanes, x)
      next_peak = tl.maximum(tl.max((x).to(tl.float32), axis=0), peak)
      weight = tl.exp(x - next_peak)
      total = total + tl.sum(weight, 0)
      peak = next_peak
    }
    tl.store(Y, total)
  }

theorem complete_source (X Copy Y : RegionName) (N B : Nat) :
    source X Copy Y N B = expanded X Copy Y N B := rfl

theorem all_executions {α : Type} [Inhabited α] (M : Algebra α)
    (X Copy Y : RegionName) (N B : Nat) (s : State α) :
    exec M (source X Copy Y N B) s = exec M (expanded X Copy Y N B) s := by
  rw [complete_source]

#axiomsClean complete_source
#axiomsClean all_executions
/-- Copies retain storage provenance; floating arithmetic replaces it even
when the source dtype is still FP32. The second max must not gain a cast. -/
def aliases (X Copy Y : RegionName) (N B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    offsets = tl.arange(0, B)
    read_ptr = X + offsets
    x = tl.load(read_ptr)
    copied = x
    peak = tl.maximum(tl.max(copied, 0), float("-inf"))
    x = x + 0.0
    second = tl.maximum(tl.max(x, 0), peak)
    tl.store(Y, second)
  }

def aliasesExpanded (X Copy Y : RegionName) (N B : Nat) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = DTYPE, Copy = DTYPE, Y = DTYPE
    offsets = tl.arange(0, $(B))
    read_ptr = X + offsets
    x = tl.load(read_ptr)
    copied = x
    peak = tl.maximum(tl.max((copied).to(tl.float32), axis=0), float("-inf"))
    x = x + 0.0
    second = tl.maximum(tl.max(x, 0), peak)
    tl.store(Y, second)
  }

theorem aliases_and_reassignment (X Copy Y : RegionName) (N B : Nat) :
    aliases X Copy Y N B = aliasesExpanded X Copy Y N B := rfl

#axiomsClean aliases_and_reassignment
end VeriTile.Tests.PositionalReductionLoop
