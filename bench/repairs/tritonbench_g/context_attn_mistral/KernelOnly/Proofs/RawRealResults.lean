import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Kernels
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Coverage

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open Checked Execution
noncomputable section

/-- The public execution body is definitionally the shared complete quote. -/
theorem shared_source (p : Formula.Layout) (r : Program.Regions) (scale : ℝ) :
    Program.kernel p r .fp32 Bool.false scale=
      kernel r.query r.key r.value scale r.start r.sequence r.out
        p.qRow p.qHead p.qDim p.kRow p.kHead p.kDim p.vRow p.vHead p.vDim
        p.outRow p.outHead p.outDim p.kvGroup p.window p.BM p.D p.BN := rfl

/-- Independent initial-array finite-sentinel weighted sum. Padded zero-V
lanes remain in the denominator; no ideal hard-mask formula is substituted. -/
abbrev realTarget := Logical.expected

/-- Actual complete quoted Real kernel over the finite grid. Only ordinary
input bindings, bounds, and allocation separation are hypotheses. -/
theorem raw_kernel_real (c : Host.Call) (allocation : FlatAlloc) (before : BlockState)
    (input : Logical.Tensors) (valid : Host.Valid c allocation before)
    (reads : Logical.Resolved c before input)
    (separate : allocation.Disjoint)
    (complete : ∀ rg,rg∉allocation.regions → allocation.extent rg=0) :
    ∃ after,Nonempty (Kernel.GridLaunchedOrdinary
      (allocation.flattenKernel (Program.kernel c.layout c.regions .fp32 Bool.false
        (Host.scale c.layout.D)).eraseDType.toAlgKernel)
      c.grid (allocation.flattenState before) after) ∧
      (∀ q : Coord c,after.readMem allocation.flat (allocation.addr c.regions.out (offset q))=
        realTarget c input (batch q) (head q) (row q) (dim q)) ∧
      BlockState.WriteWithin (c.writes allocation) (allocation.flattenState before) after ∧
      (∀ rg,rg∈c.inputs → ∀ off,off<allocation.extent rg →
        after.mem allocation.flat (allocation.addr rg off)=
          (allocation.flattenState before).mem allocation.flat (allocation.addr rg off)) := by
  obtain ⟨result,execution,output,frame,stable⟩ := Host.whole_call c allocation before valid separate complete Bool.false
  refine ⟨result.2,execution.2,?_,frame,stable⟩
  intro q
  exact (output q.val.1 q.val.2.1 (row q) q.property q.val.2.2.1.isLt (dim q)).trans
    (Logical.expected_eq c allocation before input valid reads _ _ _ _
      q.val.1.isLt q.val.2.1.isLt q.property (dim q).isLt)

#specNonCircular realTarget avoiding [stepStmts,FP.Structural.exec,KernelReference.runGrid,Mathematical.source]
#axiomsClean shared_source
#axiomsClean raw_kernel_real
#auditStmt raw_kernel_real
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel
