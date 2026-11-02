import VeriTile.Triton.DSL.KeywordBlockAdvance
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.KeywordBlockAdvance
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedVariables false

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["N","STRIDE"]
  readPorts := ["X"]
  writePorts := ["Y"]
def named (X Y : RegionName) (N STRIDE : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    ptr = tl.make_block_ptr(base=X, shape=(N, N), strides=(STRIDE, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    out = tl.make_block_ptr(base=Y, shape=(N, N), strides=(STRIDE, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    for i in range(0, 2):
        x = tl.load(ptr, boundary_check=(0, 1), padding_option="zero")
        tl.store(out, x)
        ptr = tl.advance(ptr, offsets=(1, 2))
  }
def positional (X Y : RegionName) (N STRIDE : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    ptr = tl.make_block_ptr(base=X, shape=(N, N), strides=(STRIDE, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    out = tl.make_block_ptr(base=Y, shape=(N, N), strides=(STRIDE, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    for i in range(0, 2):
        x = tl.load(ptr, boundary_check=(0, 1), padding_option="zero")
        tl.store(out, x)
        ptr = tl.advance(ptr, (1, 2))
  }
/-- Complete source AST equality covers every loop iteration, load, store,
pointer root and both distinct offsets for arbitrary parent shape/stride. -/
theorem named_offsets_preserve_program (X Y : RegionName) (N STRIDE : Nat) :
    named X Y N STRIDE=positional X Y N STRIDE := rfl

/-- error: tl.advance only accepts the offsets keyword -/
#guard_msgs in
def badKeyword (X Y : RegionName) (N STRIDE : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    ptr = tl.make_block_ptr(base=X, shape=(N, N), strides=(STRIDE, 1), offsets=(0, 0), block_shape=(1, 1), order=(1, 0))
    ptr = tl.advance(ptr, unknown=(1, 2))
  }

#axiomsClean named_offsets_preserve_program
end VeriTile.Tests.KeywordBlockAdvance
