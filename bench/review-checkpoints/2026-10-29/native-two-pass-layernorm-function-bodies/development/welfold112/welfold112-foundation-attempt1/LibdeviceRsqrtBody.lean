import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.LibdeviceRsqrtBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
def library (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = libdevice.rsqrt(x)
    tl.store(Y, y)
  }
def intrinsic (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = tl.rsqrt(x)
    tl.store(Y, y)
  }

/-- Distinct tags detect collapsing the device-library and intrinsic ports.
This deliberately does not assert a numerical property of either operation. -/
def unaryTag : UnaryOp → Nat → Nat
  | .libdeviceRsqrt, x => x+100
  | .rsqrt, x => x+200
  | _, x => x+300

def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ op x => unaryTag op x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def initial : State Nat where
  mem := fun _ _ => .mk .real 7
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem library_uses_its_own_port :
    (FP.Structural.exec model (library "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 107) := by
  simp [FP.Structural.exec,library,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,unaryTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,TileShape.allIndices,State.write,Cell.read,
    numeric,bop,load,store]
theorem intrinsic_keeps_its_port :
    (FP.Structural.exec model (intrinsic "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 207) := by
  simp [FP.Structural.exec,intrinsic,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,unaryTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,TileShape.allIndices,State.write,Cell.read,
    numeric,bop,load,store]
end
#axiomsClean library_uses_its_own_port
#axiomsClean intrinsic_keeps_its_port
end VeriTile.Tests.LibdeviceRsqrtBody
