import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.TypedDefaultDotLoopBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Y",.fp32)]

def source (Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    a = tl.full((1, 1), 1.0, tl.float16)
    b = tl.full((1, 1), 1.0, tl.float16)
    accum = tl.zeros((1, 1), tl.float32)
    for step in range(2):
        accum += tl.dot(a, b)
    lane = tl.arange(0, 1)
    tl.store(Y + lane[:, None] + lane[None, :], accum)
  }

/- The tagged algebra distinguishes an absent allow_tf32 keyword from False,
and checks actual typed-dot routing and loop accumulation;
it is not a numerical or hardware model. An untyped dot gives a different tag. -/
def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a+b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => 999)
  dotTyped := fun _ left right out options {_} {_} {_} {_} _ _ =>
    some (fun _ => ofFloat out
      (if left=.fp16 ∧ right=.fp16 ∧ out=.fp32 ∧ options.allowTF32=some Bool.false
       then 7 else 41))

def initial : State Nat where
  mem := fun _ _ => .mk .real 99
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem default_dot_loop_result :
    (FP.Structural.exec model (source "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 82) := by
  simp [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,range,loop,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,List.finRange_succ,State.write,Cell.read,numeric,bop,store,String.reduceEq]

theorem missing_typed_port_preserves_failure :
    FP.Structural.exec
      { model with dotTyped := fun _ _ _ _ _ {_} {_} {_} {_} _ _ => none }
      (source "y") initial = none := by
  simp [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,range,loop,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg]

#axiomsClean default_dot_loop_result
#axiomsClean missing_typed_port_preserves_failure
end
end VeriTile.Tests.TypedDefaultDotLoopBody
