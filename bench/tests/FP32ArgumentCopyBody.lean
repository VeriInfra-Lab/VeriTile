import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FP32ArgumentCopyBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("scale",.fp32)]

def source (Y : RegionName) (scale : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [scale := scale] in triton {
    copied = scale
    tl.store(Y, copied)
  }

/-- Precision tags expose a lost input precision or an extra output cast. -/
def model : Algebra Nat where
  literal := fun precision _ _ => if precision=some .fp32 then 7 else 99
  negInf := 0
  binary := fun _ _ _ a b => a+b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a+100
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

def initial : State Nat where
  mem := fun _ _ => .mk .real 81
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem copied_argument_keeps_fp32 (scale : ℝ) :
    (FP.Structural.exec model (source "y" scale) initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 7) := by
  simp [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,store,String.reduceEq]

#axiomsClean copied_argument_keeps_fp32
end
end VeriTile.Tests.FP32ArgumentCopyBody
