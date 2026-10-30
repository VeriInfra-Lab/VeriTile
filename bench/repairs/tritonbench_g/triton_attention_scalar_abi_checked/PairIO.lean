import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.CandidateKernels
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Pair
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section

def sourceInvocation {α : Type} (rest : Algebra α) (P : Primitives α)
    (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program) : Invocation α :=
  { PublicIO.sourceInvocation rest P kind c scale pid with
    kernel := match kind with
      | .forward => forwardKernel c scale
      | .preprocess => CandidateSource.preprocessKernel c
      | .backward => backwardKernel c scale }

def invocation (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program) :
    MaterializedInvocation Native.Graph Native.Expr :=
  { PublicIO.nativeInvocation one kind c scale owners pid with
    source := sourceInvocation (Native.algebraScalar one kind) (Native.modelScalar one kind) kind c scale pid }

def schedule (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (fw pw bw : List Grid.Program) :=
  let one := Native.constexprScale scalar
  let scale := scalar.project scalar.raw
  fw.map (invocation one .forward c scale owners) ++
    pw.map (invocation one .preprocess c scale owners) ++
      bw.map (invocation one .backward c scale owners)

/-- Task-local view for the existing materialized native pipeline. The IO,
input-cell binding, zero-DQ domain, codec and observer are fixed. The only data
is an actual list of the shared MaterializedInvocations, never a runner. -/
structure Implementation (io : MaterializedProgramIO Native.Word Native.Graph Native.Expr)
    (domain : io.Inputs → Prop) where
  schedule : io.Inputs → List (MaterializedInvocation Native.Graph Native.Expr)

def Implementation.Equivalent {io : MaterializedProgramIO Native.Word Native.Graph Native.Expr}
    {domain : io.Inputs → Prop} (left right : Implementation io domain) : Prop :=
  ∀ xs, domain xs → ∀ initial, io.InputCells xs initial →
    (∀ bounds, io.InBounds bounds xs →
      MaterializedScheduleSafe bounds (left.schedule xs) initial ∧
        MaterializedScheduleSafe bounds (right.schedule xs) initial) ∧
    ∃ l r, runMaterializedSchedule (left.schedule xs) initial=some l ∧
      runMaterializedSchedule (right.schedule xs) initial=some r ∧
      (∀ o j, io.outputMask o xs j →
        mapCell (io.observe o j) (l io.outputs[o].buf (io.write o xs j))=
          mapCell (io.observe o j) (r io.outputs[o].buf (io.write o xs j))) ∧
      io.Frame xs initial l ∧ io.Frame xs initial r

instance programSyntax {io : MaterializedProgramIO Native.Word Native.Graph Native.Expr}
    {domain : io.Inputs → Prop} : Spec.ProgramSyntax (Implementation io domain) where
  Statement := Implementation io domain
  Signature := Unit
  signature := fun _=>()
  body := fun p=>[p]
  structural := some Implementation.Equivalent
  sameContext := Eq

def original (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program) :
    Implementation (PublicIO.nativeIO scalar c owners I literal fw pw bw) (PublicIO.ZeroDQ 0 c) where
  schedule := (PublicIO.nativeIO scalar c owners I literal fw pw bw).schedule

def shifted (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program) :
    Implementation (PublicIO.nativeIO scalar c owners I literal fw pw bw) (PublicIO.ZeroDQ 0 c) where
  schedule := fun _=>schedule scalar c owners fw pw bw

end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Pair
