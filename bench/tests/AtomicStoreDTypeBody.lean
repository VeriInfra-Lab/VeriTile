import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.AtomicStoreDTypeBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def narrowSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  readPorts := ["X","Y"]
  writePorts := ["Y"]
def sameSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  readPorts := ["X","Y"]
  writePorts := ["Y"]

def narrow (X Y : RegionName) : ComputeKernel :=
  with_triton_signature narrowSignature in triton {
    destination = Y + 0
    x = tl.load(X)
    tl.atomic_add(destination, x)
  }
def same (X Y : RegionName) : ComputeKernel :=
  with_triton_signature sameSignature in triton {
    destination = Y + 0
    x = tl.load(X)
    tl.atomic_add(destination, x)
  }
def masked (X Y : RegionName) : ComputeKernel :=
  with_triton_signature narrowSignature in triton {
    destination = Y + 0
    x = tl.load(X)
    active = tl.program_id(0) < 0
    tl.atomic_add(destination, x, mask=active)
  }

/-- Operation tags distinguish a missing/duplicated cast and the atomic
primitive's storage type. This algebra does not model floating numerics. -/
def castTag : FloatDType → Nat
  | .fp16 => 100 | _ => 10
def atomicTag : FloatDType → Nat
  | .fp16 => 1000 | _ => 10000
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
  atomicAdd := fun dtype => some (fun old value => old + value + atomicTag dtype)

def initial (inputType : TileDType) (input : Value Nat inputType) : State Nat where
  mem := fun r _ => if r = "x" then .mk inputType input else .mk .fp16 11
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem implicit_cast_and_storage_atomic :
    (FP.Structural.exec model (narrow "x" "y") (initial .real 7)).map (fun s => s.mem "y" 0) =
      some (.mk .fp16 1118) := by
  simp [FP.Structural.exec,narrow,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,castTag,atomicTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,FP.Structural.atomicAdd,atomicAddition,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,numericLt,natLt]

theorem matching_dtype_has_no_added_cast :
    (FP.Structural.exec model (same "x" "y") (initial .fp16 7)).map (fun s => s.mem "y" 0) =
      some (.mk .fp16 1018) := by
  simp [FP.Structural.exec,same,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,castTag,atomicTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,FP.Structural.atomicAdd,atomicAddition,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,numericLt,natLt]

theorem false_mask_preserves_output :
    (FP.Structural.exec model (masked "x" "y") (initial .real 7)).map (fun s => s.mem "y" 0) =
      some (.mk .fp16 11) := by
  simp [FP.Structural.exec,masked,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,model,initial,castTag,atomicTag,ComputeDType.eraseDType,
    ofFloat,toFloat,Region.cast,State.setReg,FP.Structural.atomicAdd,atomicAddition,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,numericLt,natLt]

end
#axiomsClean implicit_cast_and_storage_atomic
#axiomsClean matching_dtype_has_no_added_cast
#axiomsClean false_mask_preserves_output
end VeriTile.Tests.AtomicStoreDTypeBody
