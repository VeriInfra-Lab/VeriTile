import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.IntegerLiteralAssignmentBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

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
  fromNat := fun _ n => -(n : Int)
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
  simp [FP.Structural.exec,source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

theorem inline_integer_remains_weak :
    (FP.Structural.exec model (inlineSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 9001) := by
  simp [FP.Structural.exec,inlineSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

def i32_minSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = -2147483648
    result = value / denominator
    tl.store(Y, result)
  }

theorem i32_min_conversion :
    (FP.Structural.exec model (i32_minSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (-2147483647)) := by
  simp [FP.Structural.exec,i32_minSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean i32_min_conversion

def i32_maxSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 2147483647
    result = value / denominator
    tl.store(Y, result)
  }

theorem i32_max_conversion :
    (FP.Structural.exec model (i32_maxSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (2147483648)) := by
  simp [FP.Structural.exec,i32_maxSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean i32_max_conversion

def u32_minSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 2147483648
    result = value / denominator
    tl.store(Y, result)
  }

theorem u32_min_conversion :
    (FP.Structural.exec model (u32_minSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (-2147483647)) := by
  simp [FP.Structural.exec,u32_minSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean u32_min_conversion

def u32_maxSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 4294967295
    result = value / denominator
    tl.store(Y, result)
  }

theorem u32_max_conversion :
    (FP.Structural.exec model (u32_maxSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (-4294967294)) := by
  simp [FP.Structural.exec,u32_maxSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean u32_max_conversion

def i64_above_u32Source (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 4294967296
    result = value / denominator
    tl.store(Y, result)
  }

theorem i64_above_u32_conversion :
    (FP.Structural.exec model (i64_above_u32Source "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (4294967297)) := by
  simp [FP.Structural.exec,i64_above_u32Source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean i64_above_u32_conversion

def i64_minSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = -9223372036854775808
    result = value / denominator
    tl.store(Y, result)
  }

theorem i64_min_conversion :
    (FP.Structural.exec model (i64_minSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (-9223372036854775807)) := by
  simp [FP.Structural.exec,i64_minSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean i64_min_conversion

def i64_maxSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 9223372036854775807
    result = value / denominator
    tl.store(Y, result)
  }

theorem i64_max_conversion :
    (FP.Structural.exec model (i64_maxSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (9223372036854775808)) := by
  simp [FP.Structural.exec,i64_maxSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean i64_max_conversion

def u64_minSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 9223372036854775808
    result = value / denominator
    tl.store(Y, result)
  }

theorem u64_min_conversion :
    (FP.Structural.exec model (u64_minSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (-9223372036854775807)) := by
  simp [FP.Structural.exec,u64_minSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq,integralMod]

#axiomsClean u64_min_conversion

def u64_maxSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 18446744073709551615
    result = value / denominator
    tl.store(Y, result)
  }

theorem u64_max_conversion :
    (FP.Structural.exec model (u64_maxSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real (-18446744073709551614)) := by
  simp [FP.Structural.exec,u64_maxSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq,integralMod]

#axiomsClean u64_max_conversion

def tupleSource (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    left, right = 4, -3
    result = value / left + right
    tl.store(Y, result)
  }

theorem tuple_integers_materialize :
    (FP.Structural.exec model (tupleSource "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 2) := by
  simp [FP.Structural.exec,tupleSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]

#axiomsClean tuple_integers_materialize

/-- error: integer scalar literal assignment is outside Triton's 64-bit range -/
#guard_msgs in
def belowRange (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = -9223372036854775809
    tl.store(Y, value)
  }

/-- error: integer scalar literal assignment is outside Triton's 64-bit range -/
#guard_msgs in
def aboveRange (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = 18446744073709551616
    tl.store(Y, value)
  }

#axiomsClean ordinary_integer_tensor
#axiomsClean inline_integer_remains_weak
end
end VeriTile.Tests.IntegerLiteralAssignmentBody
