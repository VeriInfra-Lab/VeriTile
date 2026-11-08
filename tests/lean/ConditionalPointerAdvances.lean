import VeriTile.Triton.DSL.ConditionalPointerAdvances
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ConditionalPointerAdvances
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

def signature : CallSignature where
  pointers := [("Input",.i64),("Output",.i64)]
  scalars := []
  constexpr := ["BASE","STEP","TRIPS","REVERSE"]
  readPorts := ["Input"]
  writePorts := ["Output"]

def source (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    for j in range(0, TRIPS):
        pointer += -STEP if REVERSE else STEP
    value = tl.load(Input)
    tl.store(pointer, value)
  }

/-- Independent guarded statement form of the same integer address operation. -/
def guarded (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    for j in range(0, TRIPS):
        if REVERSE:
            pointer -= STEP
        else:
            pointer += STEP
    value = tl.load(Input)
    tl.store(pointer, value)
  }

theorem full_program (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) :
    source Input Output BASE STEP TRIPS REVERSE = guarded Input Output BASE STEP TRIPS REVERSE := rfl

theorem full_execution {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) :
    exec M (source Input Output BASE STEP TRIPS REVERSE) s =
      exec M (guarded Input Output BASE STEP TRIPS REVERSE) s := by rw [full_program]

theorem full_safety {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) (bounds : RegionName → Nat) :
    Safety.traceSafeList M bounds (source Input Output BASE STEP TRIPS REVERSE).surfaceBody s ↔
      Safety.traceSafeList M bounds (guarded Input Output BASE STEP TRIPS REVERSE).surfaceBody s := by
  rw [full_program]

theorem forward {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 2 3 1 Bool.false) s = some t ∧
      t.mem = (s.write "Output" 5 (.mk .int ((s.mem "Input" 0).read .int))).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,FP.Structural.range,loop,numeric,bop,store,State.setReg,State.write,
    TileShape.allIndices,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

theorem reverse {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 4 3 1 Bool.true) s = some t ∧
      t.mem = (s.write "Output" 1 (.mk .int ((s.mem "Input" 0).read .int))).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,FP.Structural.range,loop,numeric,bop,store,State.setReg,State.write,
    TileShape.allIndices,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

theorem two_reverse_steps {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 8 3 2 Bool.true) s = some t ∧
      t.mem = (s.write "Output" 2 (.mk .int ((s.mem "Input" 0).read .int))).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,FP.Structural.range,loop,numeric,bop,store,State.setReg,State.write,
    TileShape.allIndices,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

theorem zero_stride {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 5 0 3 Bool.true) s = some t ∧
      t.mem = (s.write "Output" 5 (.mk .int ((s.mem "Input" 0).read .int))).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,FP.Structural.range,loop,numeric,bop,store,State.setReg,State.write,
    TileShape.allIndices,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

theorem empty_loop {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (source "Input" "Output" 7 3 0 Bool.true) s = some t ∧
      t.mem = (s.write "Output" 7 (.mk .int ((s.mem "Input" 0).read .int))).mem := by
  simp [source,exec,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,FP.Structural.range,loop,numeric,bop,store,State.setReg,State.write,
    TileShape.allIndices,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]

def floatingSignature : CallSignature := { signature with
  pointers := [("Input",.fp32),("Output",.fp32)] }

def floatingSource (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call floatingSignature call_args [] in triton {
    pointer = Output + BASE
    for j in range(0, TRIPS):
        pointer += -STEP if REVERSE else STEP
    value = tl.load(Input).to(tl.float32)
    tl.store(pointer, value)
  }

/-- An actual FP32 read/conversion and explicit cast survive the pointer
advance. This holds for arbitrary primitive interpretations, not just identity casts. -/
theorem floating_casts {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (floatingSource "Input" "Output" 4 3 1 Bool.true) s = some t ∧
      t.mem = (s.write "Output" 1 (.mk .real
        (M.cast (some .fp32) .real .real
          (M.fp32Load ((s.mem "Input" 0).read .real))))).mem := by
  simp [floatingSource,exec,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,FP.Structural.range,loop,numeric,bop,store,State.setReg,State.write,
    ComputeDType.eraseDType,TileShape.allIndices,Broadcast.leftIndex,Broadcast.rightIndex,
    Region.cast,ofFloat,toFloat]

/-- error: conditional pointer advances require the same positive and negative pitch -/
#guard_msgs in
def different_pitch (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    pointer += -STEP if REVERSE else BASE
  }

/-- error: conditional pointer pitch must have a Nat carrier -/
#guard_msgs in
def signed_pitch (Input Output : RegionName) (BASE TRIPS : Nat) (STEP : Int) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    pointer += -STEP if REVERSE else STEP
  }

/-- error: conditional pointer selector must be Boolean -/
#guard_msgs in
def non_boolean_selector (Input Output : RegionName) (BASE STEP TRIPS REVERSE : Nat) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    pointer += -STEP if REVERSE else STEP
  }

/-- error: conditional advance metadata must remain immutable -/
#guard_msgs in
def rebound_pitch (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    STEP = 0
    pointer += -STEP if REVERSE else STEP
  }

/-- error: conditional advance metadata must remain immutable -/
#guard_msgs in
def rebound_selector (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    REVERSE = False
    pointer += -STEP if REVERSE else STEP
  }

/-- error: loop shadows conditional advance metadata -/
#guard_msgs in
def shadowed_pitch (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    for STEP in range(0, TRIPS):
        pointer += -STEP if REVERSE else STEP
  }

/-- error: conditional advance metadata must remain immutable -/
#guard_msgs in
def tuple_rebound (Input Output : RegionName) (BASE STEP TRIPS : Nat) (REVERSE : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    pointer = Output + BASE
    REVERSE, saved = False, True
    pointer += -STEP if REVERSE else STEP
  }

#axiomsClean floating_casts
#axiomsClean full_program
#axiomsClean full_execution
#axiomsClean full_safety
#axiomsClean forward
#axiomsClean reverse
#axiomsClean two_reverse_steps
#axiomsClean zero_stride
#axiomsClean empty_loop
end
end VeriTile.Tests.ConditionalPointerAdvances
