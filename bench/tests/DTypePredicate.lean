import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.DTypePredicate
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def halfSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  readPorts := ["X"]
  writePorts := ["Y"]
def byteSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.f8e4)]
  readPorts := ["X"]
  writePorts := ["Y"]

def half (X Y : RegionName) : ComputeKernel :=
  with_triton_signature halfSignature in triton {
    x = tl.load(X)
    if (Y.dtype.element_ty == tl.float8e4nv):
        c = x.to(tl.float8e4nv)
    else:
        c = x.to(tl.float16)
    tl.store(Y, c)
  }
def byte (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = tl.load(X)
    if (Y.dtype.element_ty == tl.float8e4nv):
        c = x.to(tl.float8e4nv)
    else:
        c = x.to(tl.float16)
    tl.store(Y, c)
  }
def aliasHalf (X Y : RegionName) : ComputeKernel :=
  with_triton_signature halfSignature in triton {
    p = Y + 0
    x = tl.load(X)
    if p.dtype.element_ty != tl.float8e4nv:
        c = x.to(tl.float16)
    else:
        c = x.to(tl.float8e4nv)
    tl.store(Y, c)
  }

/-- An operation-counting algebra makes missing, duplicated, or wrong-format
casts observable. It is not an interpretation of floating-point numerics. -/
def castTag : FloatDType → Nat
  | .fp16 => 100 | .f8e4 => 1000 | _ => 10
def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ target value => value + castTag target
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

theorem half_executes_selected_cast :
    (FP.Structural.exec model (half "x" "y") initial).map (fun s => s.mem "y" 0) =
      some (.mk .fp16 107) := by
  simp [FP.Structural.exec,half,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,castTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,FP.Structural.store,TileShape.allIndices,State.write]

theorem byte_executes_selected_cast :
    (FP.Structural.exec model (byte "x" "y") initial).map (fun s => s.mem "y" 0) =
      some (.mk .f8e4 1007) := by
  simp [FP.Structural.exec,byte,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,castTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,FP.Structural.store,TileShape.allIndices,State.write]

theorem alias_uses_known_element_type :
    (FP.Structural.exec model (aliasHalf "x" "y") initial).map (fun s => s.mem "y" 0) =
      some (.mk .fp16 107) := by
  simp [FP.Structural.exec,aliasHalf,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,castTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,FP.Structural.store,TileShape.allIndices,State.write]

/-- error: dtype predicate requires a known pointer element or tensor value type -/
#guard_msgs in
example (X Y : RegionName) : ComputeKernel :=
  with_triton_signature halfSignature in triton {
    if unknown.dtype.element_ty == tl.float16:
        tl.store(Y, tl.load(X))
  }
end
#axiomsClean half_executes_selected_cast
#axiomsClean byte_executes_selected_cast
#axiomsClean alias_uses_known_element_type
end VeriTile.Tests.DTypePredicate
