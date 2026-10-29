import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.BarrierLoopBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32),("S",.fp32)]
def kernel (X Y S : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = x
    tl.debug_barrier()
    tl.store(S, y, None)
    tl.store(S, y, tl.program_id(0) < 1)
    for i in range(0, 2):
        y = y + x
    tl.store(Y, y)
  }

/-- A visible cast tag detects forgetting source dtype across the barrier.
The loop must carry the checked FP32 value, without a new store conversion. -/
def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a+b
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x+1000
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

def initial : State Nat where
  mem := fun _ _ => .mk .real 3
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private theorem range_two (body : List ComputeStmt) (s : State Nat) :
    range model "i" 0 2 1 body s =
      (run model body (s.setReg "i" .nat [] (fun _ => 0))).bind
        (fun t => run model body (t.setReg "i" .nat [] (fun _ => 1))) := by
  have done (t : State Nat) : range model "i" 2 2 1 body t = some t := by
    rw [range,if_neg (by decide),if_neg (by decide)]
  have once (t : State Nat) : range model "i" 1 2 1 body t =
      run model body (t.setReg "i" .nat [] (fun _ => 1)) := by
    rw [range,if_neg (by decide),if_pos (by decide)]
    simp only [show 1+1=2 from rfl,done]
    cases run model body (t.setReg "i" .nat [] (fun _ => 1)) <;> rfl
  rw [range,if_neg (by decide),if_pos (by decide)]
  simp only [Nat.zero_add,once]

theorem barrier_preserves_carried_dtype :
    (FP.Structural.exec model (kernel "x" "y" "scratch") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 9) := by
  simp [FP.Structural.exec,kernel,ComputeKernel.surfaceBody,run,step,range_two,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,numericLt,natLt]
end
#axiomsClean barrier_preserves_carried_dtype
end VeriTile.Tests.BarrierLoopBody
