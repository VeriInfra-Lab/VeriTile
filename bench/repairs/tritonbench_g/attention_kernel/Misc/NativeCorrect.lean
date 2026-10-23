import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOResults

namespace VeriTile.Bench.Repairs.AttentionKernel.FPEquiv
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
noncomputable section

/-- Native word observation of the same complete source, with the exact
FP16/BF16 storage, FP32 state, casts, dot/reduction graph, runtime scale word,
compiler routing and lane owners. `Instructions` is the original primitive
model boundary; no floating transformation or native-to-Real law is assumed. -/
theorem kernel_native (I : KernelOnly.Instructions) (p : Parameters) (bases : Bases)
    (domain : Domain p bases) (hbm : 0<p.bm) (separate : Separated p)
    (numWarps numStages : Nat) (plan : RawKernel.Plan p numWarps numStages) (scaleWord : BitVec 32)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (IO.nativeIO I (plan.layout scaleWord) p work).Implements
      (Target.native I (plan.layout scaleWord) p hbm) ∧ Closed p bases :=
  ⟨IOResults.native_implements I (plan.layout scaleWord) p bases domain hbm separate work order,
   Machine.closed p bases domain⟩

#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [KernelOnly.Instructions,Parameters,Bases,Domain,Parameters.bm,Separated,RawKernel.Plan,
   RawKernel.Plan.layout,Program,programs,grid,List.Perm,ProgramIO.Implements,
   IO.nativeIO,Target.native,Closed,FamilyNative.Expr,FamilyNative.instInhabitedExpr]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.FPEquiv
