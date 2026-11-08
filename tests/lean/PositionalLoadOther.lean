import VeriTile.Triton.DSL.PositionalLoadOther
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PositionalLoadOther
open VeriTile.Triton
noncomputable section
def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["MASK"]
  readPorts := ["X"]
  writePorts := ["Y"]
def positional (X Y : RegionName) (MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = tl.load(X, MASK, 7.0)
    tl.store(Y, x)
  }
def named (X Y : RegionName) (MASK : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    x = tl.load(X, mask=MASK, other=7.0)
    tl.store(Y, x)
  }
theorem same_complete_program (X Y : RegionName) (MASK : Bool) :
    positional X Y MASK=named X Y MASK := by cases MASK <;> rfl

theorem masked_fallback {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (X Y : RegionName) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.exec M (positional X Y Bool.false) s=some t ∧
      t.mem Y 0=.mk .real (M.fp32Load (M.literal (some .fp32) .real 7.0)) := by
  simp [positional,FP.Structural.exec,ComputeKernel.surfaceBody,FP.Structural.run,
    FP.Structural.step,FP.Structural.evalOp_unfold,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,ComputeDType.eraseDType,FP.Structural.ofFloat,FP.Structural.toFloat,
    FP.Structural.store,TileShape.allIndices,TileShape.axisDim,Option.bind]

/-- error: load mask/other was supplied both positionally and by keyword -/
#guard_msgs in
def duplicate (X Y : RegionName) : ComputeKernel :=
  let MASK := Bool.false
  with_triton_call signature call_args [] in triton {
    x = tl.load(X, MASK, 7.0, other=9.0)
    tl.store(Y, x)
  }

def halfSignature : CallSignature where
  pointers := [("X",.fp16),("Y",.fp16)]
  constexpr := ["USE_FP8"]
  readPorts := ["X"]
  writePorts := ["Y"]
/-- The inactive int8 bitcast branch must be discarded before dtype checking,
even inside a three-argument loop whose start is literal zero. -/
def closed_dtype_loop (X Y : RegionName) : ComputeKernel :=
  let USE_FP8 := Bool.false
  with_triton_call halfSignature call_args [] in triton {
    for i in range(0, 1, 1):
        x = tl.load(X, True, 0.0)
        if USE_FP8:
            x = x.to(tl.float8e5, bitcast=True)
        tl.store(Y, x)
  }
def guardSignature : CallSignature where
  pointers := [("Y",.fp16)]
  constexpr := ["CLOSED","CAUSAL"]
  writePorts := ["Y"]
def retained_guard (Y : RegionName) (CAUSAL : Bool) : ComputeKernel :=
  let CLOSED := Bool.false
  with_triton_call guardSignature call_args [] in triton {
    if CLOSED:
        tl.store(Y, 2.0)
    if CAUSAL:
        tl.store(Y, 1.0)
  }
theorem symbolic_guard_is_preserved (Y : RegionName) (CAUSAL : Bool) :
    ∃ body : List ComputeStmt, (retained_guard Y CAUSAL).surfaceBody=
      [.ifThen (.constBool CAUSAL) body] := by exact ⟨_,rfl⟩
#axiomsClean symbolic_guard_is_preserved

#axiomsClean same_complete_program
#axiomsClean masked_fallback
end
end VeriTile.Tests.PositionalLoadOther
