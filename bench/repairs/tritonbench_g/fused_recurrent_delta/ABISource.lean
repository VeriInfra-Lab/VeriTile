import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.ABI
noncomputable section
/-- The scalar records the actual FP32 launch argument after Python evaluates
an explicit positive float or K ** -0.5. No scale=1 substitution is made.
The source binder pins the Python expression and the FP32 argument bits. -/
structure ScaleArgument where
  defaulted : Bool
  launch : VeriTile.Triton.FP.Value .fp32
structure Call where
  shape : Shape
  vectorBeta : Bool
  defaultBeta : Bool
  initial : Bool
  initialRequiresGrad : Bool
  outputFinal : Bool
  terminalCotangent : Bool
  scale : ScaleArgument

def realScale (c : Call) : ℝ :=
  match c.scale.launch.toRat? with
  | some q => (q : ℝ)
  | none => 0

def flags (c : Call) : Flags :=
  ⟨c.vectorBeta,c.initial,c.outputFinal,c.initial && c.initialRequiresGrad,c.terminalCotangent,c.defaultBeta⟩
structure Tensor where
  allocation : Region
  shape : List Nat
  deriving Repr,DecidableEq
/-- All tensors are FP32, dense, on the caller's device. An allocation's
identity is its Region; distinct allocations have disjoint byte domains. -/
def tensorMeta (s : Shape) (f : Flags) (r : Region) : Tensor :=
  ⟨r,match r with
    | .q | .k | .retq | .retk => [s.B,s.H,s.T,s.K]
    | .h0 | .ht | .dht | .dh0 => [s.B,s.H,s.K,s.V]
    | .dq | .dk => [s.BNV,s.B,s.H,s.T,s.K]
    | .dbeta => if f.vectorBeta then [s.BNV,1,s.B,s.H,s.T,s.V] else [s.BNV,s.B,s.H,s.T]
    | .beta | .retbeta => if f.vectorBeta then [s.B,s.H,s.T,s.V] else [s.B,s.H,s.T]
    | .dv => [1,s.B,s.H,s.T,s.V]
    | _ => [s.B,s.H,s.T,s.V]⟩
def forwardReturn (c : Call) : Tensor×Option Tensor :=
  (tensorMeta c.shape (flags c) .o,if c.outputFinal then some (tensorMeta c.shape (flags c) .ht) else none)
def backwardReturn (c : Call) : List (Option Tensor) :=
  [some (tensorMeta c.shape (flags c) .retq),some (tensorMeta c.shape (flags c) .retk),
   some (tensorMeta c.shape (flags c) .retv),some (tensorMeta c.shape (flags c) .retbeta),none,
   if c.initial && c.initialRequiresGrad then some (tensorMeta c.shape (flags c) .dh0) else none,none]

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.ABI
