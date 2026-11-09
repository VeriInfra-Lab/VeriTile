import VeriTile.Triton.DSL.PythonConditionalAssignments
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural
namespace VeriTile.Tests.PythonConditionalAssignments
open VeriTile.Triton FP.Structural
noncomputable section

def signature : CallSignature where
  pointers := [("do",.fp32),("Out",.fp32)]
  scalars := [("T",.integer)]
  constexpr := ["N","FLAG"]
  readPorts := ["do"]
  writePorts := ["Out"]

def copied (do_ Out : RegionName) (T N : Nat) (FLAG : Bool) : ComputeKernel :=
  let «do» := do_
  with_triton_call signature call_args [T := T] in triton {
    lane = tl.arange(0, N)
    x = tl.load(do + lane)
    result = x if FLAG else tl.sum(x, 0)
    pointer = Out + lane
    for _ in range(0, T):
        tl.store(pointer, result)
        pointer += N if FLAG else 1
  }

/-- Scalar/vector constexpr choices retain their distinct types, while loop
names, pointer advances, loads and stores remain the original instructions. -/
theorem complete_program (do_ Out : RegionName) (T N : Nat) (FLAG : Bool) :
    copied do_ Out T N FLAG = if FLAG then triton {
      tl.ports inputs=[do_], outputs=[Out]
      tl.region do_ = tl.float32, Out = tl.float32
      lane = tl.arange(0, $(N))
      x = tl.load(do_ + lane)
      result = x
      pointer = Out + lane
      for «_» in range($(0), $(T), $(1)) {
        tl.store(pointer, result)
        pointer += $(N)
      }
    } else triton {
      tl.ports inputs=[do_], outputs=[Out]
      tl.region do_ = tl.float32, Out = tl.float32
      lane = tl.arange(0, $(N))
      x = tl.load(do_ + lane)
      result = tl.sum(x, 0)
      pointer = Out + lane
      for «_» in range($(0), $(T), $(1)) {
        tl.store(pointer, result)
        pointer += $(1)
      }
    } := by
  cases FLAG <;> rfl

def decrementSignature : CallSignature where
  pointers := [("X",.fp32)]
  scalars := []
  constexpr := ["N"]

def decrement (do_ : RegionName) (N : Nat) : ComputeKernel :=
  let X := do_
  with_triton_call decrementSignature call_args [] in triton {
    pointer = X + N - 1
  }

def decrementReference (do_ : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region do_ = tl.float32
  pointer = do_ + $(N) - $(1)
}

/-- Capturing the pointer prefix before decrement preserves every final
register and memory cell, for arbitrary numerical instructions and input state. -/
theorem decrement_execution {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (do_ : RegionName) (N : Nat) (state : FP.Structural.State α) :
    FP.Structural.exec M (decrement do_ N) state =
      FP.Structural.exec M (decrementReference do_ N) state := by
  simp [decrement,decrementReference,FP.Structural.exec,ComputeKernel.surfaceBody,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalComputeOp,
    FP.Structural.evalOp_unfold,Option.bind,FP.Structural.bop,Region.cast,FP.Structural.State.setReg]
  funext dtype shape name
  by_cases hn : name="pointer"
  · subst name
    by_cases hd : TileDType.ptr=dtype
    · subst dtype
      by_cases hs : shape=[]
      · subst shape
        simp only [↓reduceIte,↓reduceDIte,Option.some.injEq]
        funext i
        simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,FP.Structural.numeric]
      · simp [hs]
    · simp [hd]
  · simp [hn]

def loadSignature : CallSignature where
  pointers := [("X",.fp32)]
  scalars := []
  constexpr := ["N"]
  readPorts := ["X"]
  writePorts := []

def augmentedLoad (X : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_call loadSignature call_args [] in triton {
    h = tl.zeros([N], dtype=tl.float32)
    h += tl.load(X + tl.arange(0, N)).to(tl.float32)
  }

theorem actual_load_and_cast {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (N : Nat) (entry : State α) :
    ∃ result,exec M (augmentedLoad X N) entry=some result ∧
      result.regs .real [N] "h"=some (fun i=>M.binary (some .fp32) .real .add
        (M.literal (some .fp32) .real 0)
        (M.cast (some .fp32) .real .real
          (M.fp32Load ((entry.mem X i.1.val).read .real)))) := by
  simp [augmentedLoad,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,Region.cast,ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

/-- error: Boolean constexpr arguments must remain immutable -/
#guard_msgs in
def mutated (do_ Out : RegionName) (T N : Nat) (FLAG : Bool) : ComputeKernel :=
  let «do» := do_
  with_triton_call signature call_args [T := T] in triton {
    FLAG = False
    value = 1 if FLAG else 2
    tl.store(Out, value)
  }

#axiomsClean actual_load_and_cast
#axiomsClean complete_program
#axiomsClean decrement_execution
end
end VeriTile.Tests.PythonConditionalAssignments
