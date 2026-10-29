import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.IntegerLiteralAssignmentBody
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]

def source (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 4
    result = value / denominator
    tl.store(Y, result)
  }

def inlineSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    result = value / 4
    tl.store(Y, result)
  }

/-- Sentinels distinguish tensor integer conversion, weak floating literals,
unsigned conversion and an accidentally dropped computation precision. This
algebra is only a frontend event test, not floating-point arithmetic. -/
def model : Algebra Int where
  literal := fun _ _ _ => 9000
  negInf := 0
  binary := fun p _ _ a b => if p = some .fp32 then a+b else 5000
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => 6000+n
  fromInt := fun p n => if p = some .fp32 then n else 7000+n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def initial : State Int where
  mem := fun _ _ => .mk .real 1
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem ordinary_integer_tensor :
    (FP.Structural.exec model (source "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 5) := by
  norm_num [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store]

theorem inline_integer_remains_weak :
    (FP.Structural.exec model (inlineSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 9001) := by
  norm_num [FP.Structural.exec,inlineSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store]

#axiomsClean ordinary_integer_tensor
#axiomsClean inline_integer_remains_weak
end
end VeriTile.Tests.IntegerLiteralAssignmentBody
