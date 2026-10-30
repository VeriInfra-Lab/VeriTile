import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.OptionalStringBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["MODE"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (present chooseQK : Bool) : ComputeKernel :=
  let MODE : Option String := if present then some (if chooseQK then "qk" else "bk") else none
  with_triton_signature signature in triton {
    if MODE is None:
        ptr = X + 0
    elif MODE == "qk":
        ptr = X + 1
    elif MODE == "bk":
        ptr = X + 2
    x = tl.load(ptr)
    if MODE is not None:
        x = -x
    tl.store(Y, x)
  }

def expanded (X Y : RegionName) (present chooseQK : Bool) : ComputeKernel :=
  let absent := !present
  let isQK := present && chooseQK
  triton {
    tl.ports inputs=[X], outputs=[Y]
    tl.region X = tl.float32, Y = tl.float32
    if absent {
      ptr = X + 0
    } else {
      if isQK {
        ptr = X + 1
      } else {
        ptr = X + 2
      }
    }
    x = tl.load(ptr)
    if present {
      x = -x
    }
    tl.store(Y, x)
  }

theorem complete_program (X Y : RegionName) (present chooseQK : Bool) :
    source X Y present chooseQK = expanded X Y present chooseQK := by
  cases present <;> cases chooseQK <;> rfl

#axiomsClean complete_program

/-- error: optional string constexpr requires finite literal caller choices -/
#guard_msgs in
def opaqueChoice (MODE : Option String) : ComputeKernel :=
  with_triton_signature ({constexpr := ["MODE"]} : CallSignature) in triton {
    if MODE is None:
        x = 0
  }

/-- error: optional string constexpr may only occur in None/string comparisons -/
#guard_msgs in
def changedChoice : ComputeKernel :=
  let MODE : Option String := none
  with_triton_signature ({constexpr := ["MODE"]} : CallSignature) in triton {
    MODE = "qk"
  }
end VeriTile.Tests.OptionalStringBody
