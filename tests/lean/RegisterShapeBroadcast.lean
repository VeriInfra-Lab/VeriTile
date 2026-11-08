import VeriTile.Triton.DSL.RegisterShapeBroadcast
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.RegisterShapeBroadcast
open VeriTile.Triton
noncomputable section
def profile : CallSignature where
  pointers := [("output",.i32)]
  constexpr := ["M","N"]
def copied (output : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    mask = (r < M) & (c < N)
    value = tl.broadcast_to(r, mask.shape)
    tl.store(output + r*N+c, value, mask)
  }
def direct (output : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    r = tl.arange(0, M)[:, None]
    c = tl.arange(0, N)[None, :]
    mask = (r < M) & (c < N)
    value = tl.broadcast_to(r, [M,N])
    tl.store(output + r*N+c, value, mask)
  }
theorem receiver_shape_exact (output : RegionName) (M N : Nat) : copied output M N=direct output M N := by rfl
#axiomsClean receiver_shape_exact
/-- error: broadcast shape receiver must have a preceding assignment -/
#guard_msgs in
example (output : RegionName) (M N : Nat) : ComputeKernel :=
  with_triton_call profile call_args [] in triton {
    value = tl.broadcast_to(1, missing.shape)
    tl.store(output, value)
  }
def loopProfile : CallSignature where
  pointers := [("output",.i32)]
  constexpr := ["K","STEP","EVEN","ACC_TYPE"]
def reverseOriginal (output : RegionName) (K STEP : Nat) (EVEN : Bool) : ComputeKernel :=
  let ACC_TYPE : SourceDType := .i32
  with_triton_call loopProfile call_args [] in triton {
    acc = tl.zeros([1], dtype=ACC_TYPE)
    for k in range(K, 0, -STEP):
        if EVEN:
            acc = acc + k
        else:
            acc = acc - k
    tl.store(output + tl.arange(0, 1), acc)
  }
def directLoopProfile : CallSignature where
  pointers := [("output",.i32)]
  constexpr := ["K","STEP","EVEN"]
def reverseDirect (output : RegionName) (K STEP : Nat) (EVEN : Bool) : ComputeKernel :=
  with_triton_call directLoopProfile call_args [] in triton {
    acc = tl.zeros([1], dtype=tl.int32)
    for k in range(K, -0, -STEP):
        if EVEN:
            acc = acc + k
        else:
            acc = acc - k
    tl.store(output + tl.arange(0, 1), acc)
  }
theorem reverse_integer_dtype_exact (output : RegionName) (K STEP : Nat) (EVEN : Bool) :
    reverseOriginal output K STEP EVEN=reverseDirect output K STEP EVEN := by rfl
#axiomsClean reverse_integer_dtype_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.RegisterShapeBroadcast
