import VeriTile.Triton.DSL.NestedPointerDType
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NestedPointerDType
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false
inductive Storage where | half | float8 deriving DecidableEq

def signature (storage : Storage) : CallSignature where
  pointers := [("Out",match storage with | .half => .fp16 | .float8 => .f8e4)]
  constexpr := ["TRIPS"]
  writePorts := ["Out"]

def source (storage : Storage) (Out : RegionName) (TRIPS : Nat) : ComputeKernel :=
  with_triton_call (signature storage) call_args [] in triton {
    for i in range(0, TRIPS):
        value = 7.0
        if Out.dtype.element_ty == tl.float8e4nv:
            converted = value.to(tl.float8e4nv)
        else:
            converted = value.to(tl.float16)
        tl.store(Out + i, converted)
  }

/-- Explicit branch selection used only as a semantic regression oracle. -/
def selected (storage : Storage) (Out : RegionName) (TRIPS : Nat) : ComputeKernel :=
  match storage with
  | .half => with_triton_call (signature .half) call_args [] in triton {
      for i in range(0, TRIPS):
          value = 7.0
          converted = value.to(tl.float16)
          tl.store(Out + i, converted)
    }
  | .float8 => with_triton_call (signature .float8) call_args [] in triton {
      for i in range(0, TRIPS):
          value = 7.0
          converted = value.to(tl.float8e4nv)
          tl.store(Out + i, converted)
    }

theorem complete_kernel (storage : Storage) (Out : RegionName) (TRIPS : Nat) :
    source storage Out TRIPS = selected storage Out TRIPS := by cases storage <;> rfl

theorem complete_execution {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (Out : RegionName) (TRIPS : Nat) (s : State α) :
    exec M (source storage Out TRIPS) s = exec M (selected storage Out TRIPS) s := by
  rw [complete_kernel]

theorem complete_safety {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (Out : RegionName) (TRIPS : Nat) (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (source storage Out TRIPS).surfaceBody s ↔
      Safety.traceSafeList M bounds (selected storage Out TRIPS).surfaceBody s := by
  rw [complete_kernel]

theorem successful_half_store {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source .half "Out" 1) s = some t ∧
      t.mem = (s.write "Out" 0 (.mk .fp16 (M.cast none .real .fp16 (M.literal none .real 7.0)))).mem := by
  simp [source,exec,run,step,loop,FP.Structural.range,evalExpr,evalOp_unfold,Option.bind,
    State.setReg,store,TileShape.allIndices,numeric,bop,Region.cast,State.write,
    evalOffsets,ofFloat,toFloat]

theorem successful_float8_store {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source .float8 "Out" 1) s = some t ∧
      t.mem = (s.write "Out" 0 (.mk .f8e4 (M.cast none .real .f8e4 (M.literal none .real 7.0)))).mem := by
  simp [source,exec,run,step,loop,FP.Structural.range,evalExpr,evalOp_unfold,Option.bind,
    State.setReg,store,TileShape.allIndices,numeric,bop,Region.cast,State.write,
    evalOffsets,ofFloat,toFloat]

/-- error: queried dtype requires an immutable pointer parameter -/
#guard_msgs in
def rebound (Out : RegionName) : ComputeKernel :=
  with_triton_call (signature .half) call_args [] in triton {
    Out = Out + 1
    if Out.dtype.element_ty == tl.float8e4nv:
        tl.store(Out, 1.0)
  }

/-- error: loop shadows a queried pointer parameter -/
#guard_msgs in
def shadowed (Out : RegionName) : ComputeKernel :=
  with_triton_call (signature .half) call_args [] in triton {
    for Out in range(0, 2):
        if Out.dtype.element_ty == tl.float8e4nv:
            value = 1.0
  }

/-- error: unsupported or undeclared pointer dtype query -/
#guard_msgs in
def unknown (Out : RegionName) : ComputeKernel :=
  with_triton_call (signature .half) call_args [] in triton {
    if Other.dtype.element_ty == tl.float8e4nv:
        value = 1.0
  }

#axiomsClean successful_half_store
#axiomsClean successful_float8_store
#axiomsClean complete_kernel
#axiomsClean complete_execution
#axiomsClean complete_safety
end
end VeriTile.Tests.NestedPointerDType
