import bench.tests.IntegerLiteralAssignmentBody

namespace VeriTile.Tests.NarrowUnsignedLoadBody
open VeriTile.Triton FP.Structural
open IntegerLiteralAssignmentBody (model)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def byteSignature : CallSignature where
  pointers := [("X",.u8),("Y",.u8)]
def halfSignature : CallSignature where
  pointers := [("X",.u16),("Y",.u16)]

def byteSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = tl.load(X)
    exponent = x - 1 + 127
    tl.store(Y, exponent)
  }
def halfSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature halfSignature in triton {
    alias = X + 0
    x = tl.load(alias)
    value = x - 1 + 2
    tl.store(Y, value)
  }
def mixedSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ byteSignature with pointers := [("X",.u8),("Y",.i64)] } : CallSignature) in triton {
    x = tl.load(X)
    one = 1
    value = x - one
    tl.store(Y, value.to(tl.int64))
  }
def initial : State Int where
  mem := fun _ _ => .mk .nat 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem byte_subtraction_wraps :
    (FP.Structural.exec model (byteSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .nat 126) := by
  simp [FP.Structural.exec,byteSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq,integralMod]
#axiomsClean byte_subtraction_wraps

theorem halfword_alias_subtraction_wraps :
    (FP.Structural.exec model (halfSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .nat 1) := by
  simp [FP.Structural.exec,halfSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq,integralMod]
#axiomsClean halfword_alias_subtraction_wraps

theorem tensor_operand_promotes_signed :
    (FP.Structural.exec model (mixedSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .int (-1)) := by
  simp [FP.Structural.exec,mixedSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq,integralMod]
#axiomsClean tensor_operand_promotes_signed

/-- error: integer scalar is outside the unsigned tensor's source dtype range -/
#guard_msgs in
def oversized (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = tl.load(X)
    value = x + 256
    tl.store(Y, value)
  }
/-- error: integer scalar is outside the unsigned tensor's source dtype range -/
#guard_msgs in
def negativeScalar (X Y : RegionName) : ComputeKernel :=
  with_triton_signature byteSignature in triton {
    x = tl.load(X)
    value = x + (-1)
    tl.store(Y, value)
  }
end
end VeriTile.Tests.NarrowUnsignedLoadBody
