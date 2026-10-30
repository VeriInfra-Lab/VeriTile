import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.Execution
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Native
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Checked
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.RealReference

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

def reductionInputs {α : Type} {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values α .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (j : Fin (TileShape.axisDim shape ax)) : α :=
  match keep with
  | Bool.false => xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true => xs (TileShape.replaceAxisIndex shape ax i j)

namespace Real
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .mul => x*y | .sub => x-y | .div => x/y | .max => max x y | .pow => x
  unary := fun _ op x => match op with
    | .exp => Real.exp x
    | _ => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {shape} ax keep xs i => ∑ j,reductionInputs ax keep xs i j

theorem source_ops : Source.ops algebra=realOps := rfl

/-- Independent exact per-shard formula. The exported kernel writes these
partials; the subsequent Torch sum is outside the new kernel proof. -/
def partialFormula (c : Shape) (x : Input ℝ) (scale : ℝ) (z ik n f : Nat) : ℝ :=
  ∑ j : Fin c.BK, if ik*c.BK+j.val<c.K then
    (realState c x z (ik*c.BK+j.val) f n+
      x.k z (visit c n) (ik*c.BK+j.val)*x.v z (visit c n) f*x.u z (ik*c.BK+j.val))*
      (x.q z (visit c n) (ik*c.BK+j.val)*scale) else 0

theorem partial_scalar (c : Shape) (x : Input ℝ) (scale : ℝ) (z ik n f : Nat) (hf : f<c.V) :
    (∑ j : Fin c.BK,scalarTerm (Source.ops algebra) c x scale z n (ik*c.BK+j.val) f)=
      partialFormula c x scale z ik n f := by
  simp only [source_ops,scalar_real_term c x scale z n _ f hf,partialFormula]
end Real

namespace Native
abbrev Term := FusedRwkv6Kernel.Native.Term
abbrev Instructions := FusedRwkv6Kernel.Native.Instructions
abbrev Compiler := FusedRwkv6Kernel.Native.Compiler
abbrev observe := FusedRwkv6Kernel.Native.observe
instance : Inhabited Term := ⟨.zero⟩

/-- A reduction choice is structural, made before values arrive. Its leaves
include padded lanes; finite/NaN/Inf classification never selects the tree. -/
structure ReductionPlan where
  tree : Nat → Tree
  legal : ∀ n,(tree n).legal n

def algebra (plan : ReductionPlan) : Algebra Term where
  literal := fun _ _ _ => .zero
  negInf := .zero
  binary := fun _ _ op x y => match op with
    | .add => .add x y
    | .mul => .mul x y
    | _ => .zero
  unary := fun _ op x => match op with
    | .exp => .exp x
    | _ => x
  cast := fun _ _ _ x => .store32 x
  fromNat := fun _ _ => .zero
  fromInt := fun _ _ => .zero
  fp32Bits := fun x => .word .outside 0 x.bits
  fp32Load := .load32
  reduceMax := fun _ {_} _ _ _ _ => .zero
  reduceSum := fun _ {shape} ax keep xs i =>
    (plan.tree (TileShape.axisDim shape ax)).eval .zero .add (fun j =>
      if h : j<TileShape.axisDim shape ax then reductionInputs ax keep xs i ⟨j,h⟩ else .zero)

/-- Scalar leaves are the observed post-binding arithmetic operands. The Python
candidate preserves the original raw argument; this port records the ordinary
fp32/i32/i64/u64 ABI and arithmetic conversion, not a numerical experiment. -/
structure ScalarBinding where
  toFP32 : FusedRwkv6Kernel.Checked.PythonNumber → BitVec 32

def scalar (binding : ScalarBinding) (raw : FusedRwkv6Kernel.Checked.PythonNumber) : Term :=
  .word .outside 0 (binding.toFP32 raw)

def plans (c : Shape) (p : ReductionPlan) : Plans c where
  key := fun _ _ _ _ => p.tree c.BK
  host := fun _ _ _ => Tree.chain c.NK
  keyLegal := fun _ _ _ _ => p.legal c.BK
  hostLegal := fun _ _ _ => Tree.chain_legal c.NK

end Native
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked
