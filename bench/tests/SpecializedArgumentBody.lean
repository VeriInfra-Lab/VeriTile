import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SpecializedArgumentBody
open VeriTile.Triton

def scalarSignature : CallSignature where
  scalars := [("VALUE",.integer)]
  constexpr := ["IS_POINTER"]
  specializeBooleans := ["IS_POINTER"]

def pointerSignature : CallSignature where
  pointers := [("VALUE",.i32)]
  constexpr := ["IS_POINTER"]
  specializeBooleans := ["IS_POINTER"]

def scalarSource (v : Int) : ComputeKernel :=
  let IS_POINTER : Bool := Bool.false
  with_triton_call scalarSignature call_args [VALUE := v] in triton {
    if IS_POINTER:
        x = tl.load(VALUE)
    else:
        x = VALUE
    if x < 0:
        return
    y = x + 2
  }

def pointerSource (VALUE : RegionName) : ComputeKernel :=
  let IS_POINTER : Bool := Bool.true
  with_triton_signature pointerSignature in triton {
    if IS_POINTER:
        x = tl.load(VALUE)
    else:
        x = VALUE
    if x < 0:
        return
    y = x + 2
  }

def scalarExpected (v : Int) : ComputeKernel :=
  with_triton_call { scalarSignature with constexpr := [], specializeBooleans := [] }
      call_args [VALUE := v] in triton {
    x = VALUE
    if x < 0:
        return
    y = x + 2
  }

def pointerExpected (VALUE : RegionName) : ComputeKernel :=
  with_triton_signature { pointerSignature with constexpr := [], specializeBooleans := [] } in triton {
    x = tl.load(VALUE)
    if x < 0:
        return
    y = x + 2
  }

theorem scalar_complete (v : Int) : scalarSource v = scalarExpected v := rfl
theorem pointer_complete (VALUE : RegionName) : pointerSource VALUE = pointerExpected VALUE := rfl

/-- error: explicit Boolean specialization requires a closed caller value -/
#guard_msgs in
def unknownCase (IS_POINTER : Bool) (VALUE : RegionName) : ComputeKernel :=
  with_triton_signature pointerSignature in triton {
    x = tl.load(VALUE)
  }

/-- error: pointer types belong in the call signature, outside the pasted body -/
#guard_msgs in
def metadataInDiscardedBranch (VALUE : RegionName) : ComputeKernel :=
  let IS_POINTER : Bool := Bool.true
  with_triton_signature pointerSignature in triton {
    if IS_POINTER:
        x = tl.load(VALUE)
    else:
        tl.region VALUE = tl.int32
  }

#axiomsClean scalar_complete
#axiomsClean pointer_complete
end VeriTile.Tests.SpecializedArgumentBody
