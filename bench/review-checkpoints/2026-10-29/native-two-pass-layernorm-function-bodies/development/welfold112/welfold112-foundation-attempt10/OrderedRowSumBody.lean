import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.OrderedRowSumBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def signature (tree : DSL.WelfordSource.Tree 4) : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  orderedRowSums := [("values",⟨4,tree⟩)]

def kernel (tree : DSL.WelfordSource.Tree 4) (X Y : RegionName) : ComputeKernel :=
  with_triton_signature (signature tree) in triton {
    offsets = tl.arange(0, 4)[None, :]
    values = tl.load(X + offsets)
    total = tl.sum(values, 1)
    tl.store(Y + tl.arange(0, 1), total)
  }

def leftTree : DSL.WelfordSource.Tree 4 :=
  .node (.node (.node (.leaf 0) (.leaf 1)) (.leaf 2)) (.leaf 3)
def rightTree : DSL.WelfordSource.Tree 4 :=
  .node (.leaf 0) (.node (.leaf 1) (.node (.leaf 2) (.leaf 3)))

/-- A deliberately nonassociative operation records tree order. The ordinary
reduceSum sentinel detects ignoring the explicit metadata. This is a frontend
regression algebra, not a model of floating-point numeric values. -/
def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a * 100 + b
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 999999

def initial : State Nat where
  mem := fun r o => .mk .real (if r = "x" then o+1 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem left_order :
    (FP.Structural.exec model (kernel leftTree "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 1020304) := by
  simp [FP.Structural.exec,kernel,leftTree,DSL.WelfordSource.sumExpression,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,model,initial,ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,
    State.setReg,TileShape.allIndices,List.finRange_succ,State.write,Cell.read,
    numeric,bop,numericLt,natLt,store]

theorem right_order :
    (FP.Structural.exec model (kernel rightTree "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 604) := by
  simp [FP.Structural.exec,kernel,rightTree,DSL.WelfordSource.sumExpression,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,model,initial,ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,
    State.setReg,TileShape.allIndices,List.finRange_succ,State.write,Cell.read,
    numeric,bop,numericLt,natLt,store]

/-- Follow the raw LayerNorm spelling through both internal reshape and the
source axis insertion. Untyped division/fromNat produce distinct sentinels. -/
def dividedKernel (tree : DSL.WelfordSource.Tree 4) (X Y : RegionName) : ComputeKernel :=
  with_triton_signature (signature tree) in triton {
    offsets = tl.arange(0, 4)[None, :]
    values = tl.load(X + offsets)
    total = tl.sum(values, 1)[:, None]
    denominator = 4
    mean = total / denominator
    tl.store(Y + tl.arange(0, 1)[:, None], mean)
  }

def precisionModel : Algebra Nat := { model with
  binary := fun p _ op a b =>
    if p = some .fp32 then
      match op with | .div => a+b | _ => a*100+b
    else 888888
  fromNat := fun p n => if p = some .fp32 then n else 777777 }

theorem division_precision :
    (FP.Structural.exec precisionModel (dividedKernel leftTree "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 1020308) := by
  simp [FP.Structural.exec,dividedKernel,leftTree,DSL.WelfordSource.sumExpression,
    ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,precisionModel,model,initial,ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,
    State.setReg,TileShape.allIndices,List.finRange_succ,State.write,Cell.read,
    numeric,bop,numericLt,natLt,store]

end
#axiomsClean division_precision
#axiomsClean left_order
#axiomsClean right_order
end VeriTile.Tests.OrderedRowSumBody
