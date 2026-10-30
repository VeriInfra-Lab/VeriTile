import bench.tritonbench_g.layer_norm_fwd.Kernels

/-! Complete source-only, width-erased repaired kernel. The imported macro is
also source-only and retains every cast, branch, Philox round and masked store.
The Python repair widens program_id to signed64 before pointer products; exact
machine-width addressing is connected in Proofs.NativeAccess and NativeABI.
Destination dtype specialization remains explicit in layerNormFwdSource%.
-/
namespace VeriTile.Bench.Repairs.LayerNormFwd.Source
open VeriTile.Triton

structure Flags where
  rms : Bool
  residual : Bool
  residualOut : Bool
  bias : Bool
  dropout : Bool
  dropoutMask : Bool
  rowscale : Bool
  x1 : Bool
  w1 : Bool
  b1 : Bool
  deriving DecidableEq

structure Regions where
  x : RegionName
  y : RegionName
  weight : RegionName
  bias : RegionName
  residual : RegionName
  x1 : RegionName
  weight1 : RegionName
  bias1 : RegionName
  y1 : RegionName
  residualOut : RegionName
  rowscale : RegionName
  seeds : RegionName
  dropoutMask : RegionName
  mean : RegionName
  rstd : RegionName

structure Strides where
  x : Nat
  y : Nat
  residual : Nat
  residualOut : Nat
  x1 : Nat
  y1 : Nat

abbrev sourceType := Flags → Regions → Strides → Nat → Nat → ℝ → ℝ →
  (block : Nat) → Random.ArangeSizeDomain block → ComputeKernel

def kernel : sourceType :=
  fun f r stride M N eps p block hBlock =>
    (layerNormFwdSource%[tl.float32, tl.float32, tl.float32, tl.float32,
      tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, original])
    r.x r.y r.weight r.bias r.residual r.x1 r.weight1 r.bias1 r.y1
    r.residualOut r.rowscale r.seeds.cast r.dropoutMask.cast r.mean r.rstd
    stride.x stride.y stride.residual stride.residualOut stride.x1 stride.y1
    M N eps p f.rms block hBlock f.residual f.residualOut f.bias f.dropout
    f.dropoutMask f.rowscale f.x1 f.w1 f.b1

def rowBits : Nat := 64
def scalarBits : Nat := 32
def seedStorageBits : Nat := 64
def randomRounds : Nat := 7
end VeriTile.Bench.Repairs.LayerNormFwd.Source

/-! Native ordered reduction syntax for the whole-host extension. Arithmetic
contraction is structural: a square-plus-add is represented by one FMA node.
No numerical premise equates it to a separately rounded multiply and add. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section

inductive SumTree (block : Nat) where
  | leaf (lane : Fin block)
  | add (left right : SumTree block)

def SumTree.leaves : SumTree block → List (Fin block)
  | .leaf j => [j]
  | .add a b => a.leaves ++ b.leaves

def SumTree.eval (add : α → α → α) (x : Fin block → α) : SumTree block → α
  | .leaf j => x j
  | .add a b => add (a.eval add x) (b.eval add x)

/-- Every leaf contributes its square once. The fused constructor explicitly
places the squared lane first and the accumulated subtree last, preserving
operand order even for signed zeros, infinities and NaN payloads. -/
inductive SquareTree (block : Nat) where
  | square (lane : Fin block)
  | add (left right : SquareTree block)
  | fmaSquare (lane : Fin block) (rest : SquareTree block)

def SquareTree.leaves : SquareTree block → List (Fin block)
  | .square j => [j]
  | .add a b => a.leaves ++ b.leaves
  | .fmaSquare j b => j :: b.leaves

def SquareTree.eval (mul add : α → α → α) (fma : α → α → α → α)
    (x : Fin block → α) : SquareTree block → α
  | .square j => mul (x j) (x j)
  | .add a b => add (a.eval mul add fma x) (b.eval mul add fma x)
  | .fmaSquare j b => fma (x j) (x j) (b.eval mul add fma x)

structure Reductions (block : Nat) where
  mean : SumTree block
  variance : SquareTree block
  meanComplete : mean.leaves.Perm (List.finRange block)
  varianceComplete : variance.leaves.Perm (List.finRange block)

/-- Scalar leaves only. Reductions, masks, pointers, random state evolution,
host choices and output construction are deliberately absent from this API. -/
structure Numerics (α : Type) where
  literal : ℝ → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  div : α → α → α
  fma : α → α → α → α
  sqrt : α → α
  greater : α → α → Bool
  randomValue : VeriTile.Triton.FP.Value .fp32 → α

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
