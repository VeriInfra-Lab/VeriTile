import bench.tritonbench_g.triton_linear_activation.Proofs.Support.SourceKernels

namespace VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
/-- A proof projection instantiates the same copied body with explicit flags;
there is no separate implementation. The source_split equality reconnects all
three actual flags, including their full branches and loop body. -/
def flagged (mode : Mode) (p : Parameters) (bias save masked : Bool) : ComputeKernel :=
  typed (dtype mode) p.output p.saved p.input p.weight p.bias
    p.rows p.columns p.inner p.outputRowStride p.outputColStride p.savedRowStride p.savedColStride
    p.inputRowStride p.inputInnerStride p.weightColStride p.weightInnerStride
    p.blockRows p.groupRows p.blockColumns p.blockInner bias save masked p.activation

def frontCount (p : Parameters) : Nat := 16 + if p.hasBias then 3 else 0

def front (mode : Mode) (p : Parameters) := (flagged mode p p.hasBias Bool.false Bool.false).surfaceBody.take (frontCount p)
def statement (mode : Mode) (p : Parameters) : ComputeStmt :=
  ((flagged mode p Bool.false Bool.false (p.inner%p.blockInner==0)).surfaceBody[16]?).getD (.effectMarker "missing loop")
def body (mode : Mode) (p : Parameters) : List ComputeStmt :=
  match statement mode p with
  | .forRangeInt _ _ _ _ body => body
  | _ => []
def tailCode (mode : Mode) (p : Parameters) := (flagged mode p Bool.false p.saveInputs Bool.false).surfaceBody.drop 17

theorem source_split (mode : Mode) (p : Parameters) :
    (source mode p).surfaceBody=front mode p++[statement mode p]++tailCode mode p := by
  cases mode <;> cases hb : p.hasBias <;> cases hs : p.saveInputs <;>
    cases hl : (p.inner%p.blockInner==0) <;>
    simp only [front,statement,tailCode,frontCount,source,flagged,typed,originalKernel,
      dtype,hb,hs,hl,Bool.false_eq_true,if_false,if_true] <;> rfl

theorem source_statements (mode : Mode) (p : Parameters) :
    (source mode p).surfaceBody.length=24+(if p.hasBias then 3 else 0)+(if p.saveInputs then 3 else 0) := by
  cases mode <;> cases hb : p.hasBias <;> cases hs : p.saveInputs <;>
    cases hl : (p.inner%p.blockInner==0) <;>
    simp only [source,typed,originalKernel,dtype,hb,hs,hl,
      Bool.false_eq_true,if_false,if_true] <;> rfl

#axiomsClean source_split
#axiomsClean source_statements
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.TritonLinearActivation.KernelSource
