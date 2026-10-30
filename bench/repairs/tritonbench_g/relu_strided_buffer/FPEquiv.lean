import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile Triton FP.Structural
open TritonBenchG.ReluStridedBuffer.Whole
open scoped VeriTile.Spec
noncomputable section
local instance wordInhabited : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

def assumptions : Spec.Assumptions ComputeStmt := []

def original (rest : Algebra (FP.Value .fp32)) (c : Parameters) :
    FP.BackendMemory.Program (fp32Backend rest) := ⟨kernel c⟩

def folded (rest : Algebra (FP.Value .fp32)) (c : Parameters) :
    FP.BackendMemory.Program (fp32Backend rest) := ⟨foldedKernel c⟩

/-- Both complete FP32 sources succeed and agree on every memory cell for
arbitrary parameters and initial memory, including empty extents and either
branch. Only the unused integer tile count changes. No view validity,
finiteness, positive-extent or floating rewrite assumption is introduced. -/
specification relu_strided_buffer_folded_equiv (rest : Algebra (FP.Value .fp32))
    (c : Parameters) : original rest c ≡[assumptions] folded rest c :=
  SourceEquivalence.source_equivalence rest c

#print_fp_assumptions relu_strided_buffer_folded_equiv
#stmtSurfaceSubset relu_strided_buffer_folded_equiv ⊆
  [FP.Structural.Algebra,FP.Value,FP.Format.fp32,Parameters,original,folded,
   assumptions,Spec.Equiv,FP.BackendMemory.Program,FP.BackendMemory.programSyntax,
   fp32Backend,wordInhabited]
#axiomsClean relu_strided_buffer_folded_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
