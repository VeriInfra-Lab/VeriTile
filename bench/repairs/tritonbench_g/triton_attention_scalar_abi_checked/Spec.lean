import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.RealTensors
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativePipeline

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

/-- Row-major inverse, used only to present finite logical inputs to the
existing independent attention/VJP and ordered native formulas. -/
def index (c : Shape) (o : Nat) (h : o<c.elements) : Index c :=
  (⟨o/c.dim/c.rows,(Nat.div_lt_iff_lt_mul c.rows_pos).mpr
    ((Nat.div_lt_iff_lt_mul c.dim_pos).mpr h)⟩,
   ⟨o/c.dim%c.rows,Nat.mod_lt _ c.rows_pos⟩,
   ⟨o%c.dim,Nat.mod_lt _ c.dim_pos⟩,PUnit.unit)

def inputAt {α : Type} (zero : α) (c : Shape) (xs : Inputs α c)
    (r : RegionName) (o : Nat) : α :=
  if h : o<c.elements then
    let i := index c o h
    if r="Q" then xs 0 i else if r="K" then xs 1 i else if r="V" then xs 2 i
    else if r="DO" then xs 3 i else if r="DQ" then xs 4 i else zero
  else zero

/-- The real target consists exactly of causal softmax attention and its Q/K/V
vector-Jacobian product. It contains no source run or saved-buffer recurrence. -/
def realFormula (c : Shape) (scale : ℝ) (xs : Inputs ℝ c) : Outputs ℝ c :=
  let μ := inputAt 0 c xs
  fun port=>match port with
  | ⟨0,_⟩ => fun i=>Math.attention (Erased.headTensor c μ "Q" i.1)
      (Erased.headTensor c μ "K" i.1) (Erased.headTensor c μ "V" i.1) scale i.2.1 i.2.2.1
  | ⟨1,_⟩ => fun i=>Math.gradientQ (Erased.headTensor c μ "Q" i.1)
      (Erased.headTensor c μ "K" i.1) (Erased.headTensor c μ "V" i.1)
      (Erased.headTensor c μ "DO" i.1) scale i.2.1 i.2.2.1
  | ⟨2,_⟩ => fun i=>Math.gradientK (Erased.headTensor c μ "Q" i.1)
      (Erased.headTensor c μ "K" i.1) (Erased.headTensor c μ "V" i.1)
      (Erased.headTensor c μ "DO" i.1) scale i.2.1 i.2.2.1
  | ⟨3,_⟩ => fun i=>Math.gradientV (Erased.headTensor c μ "Q" i.1)
      (Erased.headTensor c μ "K" i.1) (Erased.headTensor c μ "DO" i.1) scale i.2.1 i.2.2.1

/-- The original independent ordered native formulas: actual FP32 words,
FP16 narrowing/widening, reduction replicas, instruction model and scalar
class specialization. No native-equals-real claim is introduced. -/
def nativeFormula (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (xs : Inputs Native.Word c) : Outputs Native.Word c :=
  let one := Native.constexprScale scalar
  let scale := scalar.project scalar.raw
  let x := Native.Selected.wordInput c (inputAt 0 c xs)
  let prepared := Native.Target.prepared c one scale x owners
  fun port=>match port with
  | ⟨0,_⟩ => fun i=>((Native.Target.forward c one scale x owners).output i.1.val i.2.1.val i.2.2.1.val).eval I literal
  | ⟨1,_⟩ => fun i=>(Native.Target.query c one scale prepared i.1.val i.2.1.val i.2.2.1.val c.blocks).eval I literal
  | ⟨2,_⟩ => fun i=>(Native.Target.key c one scale prepared i.1.val i.2.1.val i.2.2.1.val).eval I literal
  | ⟨3,_⟩ => fun i=>(Native.Target.value c one scale prepared i.1.val i.2.1.val i.2.2.1.val).eval I literal

#specNonCircular realFormula avoiding [FP.Structural.exec,Grid.wholeRun,Grid.wholeReference]
#specNonCircular nativeFormula avoiding [FP.Structural.exec,Native.Selected.run,Native.Selected.stores]
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
