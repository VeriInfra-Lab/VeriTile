import bench.tests.IntegerLiteralAssignmentBody

namespace VeriTile.Tests.SourceHelperLiteralBody
open VeriTile.Triton FP.Structural
open IntegerLiteralAssignmentBody (model initial)
noncomputable section
set_option linter.unusedSimpArgs false

def dividing : SourceFunction := triton_function divide_value(value, denominator) in triton {
  result = value / denominator
  return result
}

def rebinding : SourceFunction := triton_function rebind_value(value, denominator) in triton {
  denominator = denominator + 1
  result = value / denominator
  return result
}

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  functions := [dividing,rebinding]

def literalCall (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    result = divide_value(value, 4)
    tl.store(Y, result)
  }

def namedCall (X Y : RegionName) : ComputeKernel :=
  let DENOMINATOR : Nat := 4
  with_triton_signature ({ signature with constexpr := ["DENOMINATOR"] } : CallSignature) in triton {
    value = tl.load(X)
    result = divide_value(value, DENOMINATOR)
    tl.store(Y, result)
  }

def reboundCall (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    result = rebind_value(value, 4)
    tl.store(Y, result)
  }

def tensorCall (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(X)
    denominator = 4
    result = divide_value(value, denominator)
    tl.store(Y, result)
  }

/- The tagged algebra distinguishes a weak constant at the arithmetic site
from a prematurely materialized tensor; it makes no numerical FP claim. -/
theorem literal_argument_remains_weak :
    (FP.Structural.exec model (literalCall "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 9001) := by
  simp [FP.Structural.exec,literalCall,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean literal_argument_remains_weak

theorem caller_constexpr_remains_weak :
    (FP.Structural.exec model (namedCall "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 9001) := by
  simp [FP.Structural.exec,namedCall,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean caller_constexpr_remains_weak

theorem helper_assignment_materializes_tensor :
    (FP.Structural.exec model (reboundCall "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 6) := by
  simp [FP.Structural.exec,reboundCall,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean helper_assignment_materializes_tensor

theorem caller_tensor_remains_tensor :
    (FP.Structural.exec model (tensorCall "x" "y") initial).map
      (fun s => s.mem "y" 0) = some (.mk .real 5) := by
  simp [FP.Structural.exec,tensorCall,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,Option.map,model,initial,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,State.setReg,
    TileShape.allIndices,State.write,Cell.read,numeric,bop,store,String.reduceEq]
#axiomsClean caller_tensor_remains_tensor

end
end VeriTile.Tests.SourceHelperLiteralBody
