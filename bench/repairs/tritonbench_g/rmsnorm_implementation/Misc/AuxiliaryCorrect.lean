import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.IOResults

/-! Auxiliary Real correctness of the integer-index candidate. The declaration
retains its historical namespace and complete active-output contract. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.Correct
open VeriTile Triton Interface
noncomputable section

theorem rmsnorm_index_commuted_correctness (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ)
    (unique : OutputUnique n sok) :
    Spec.Real ((rowIO Bool.true sxb sxm sxk sw sob som sok n eps).RegionalScheduledImplements
      oneProgram (fun v _ k => formula eps (v (0 : Fin 2)) (v (1 : Fin 2)) k)) :=
  implements Bool.true sxb sxm sxk sw sob som sok n eps unique

#specNonCircular formula avoiding [source,rowIO,kernel,indexCommutedKernel]
#axiomsClean rmsnorm_index_commuted_correctness
#stmtSurfaceSubset rmsnorm_index_commuted_correctness ⊆
  [Spec.Real,rowIO,oneProgram,formula,OutputUnique,UKernelIO.RegionalScheduledImplements,
   UKernelIO.Ctx,UKernelIO.iarity,UKernelIO.nIn,UKernelIO.nOut,UKernelIO.oarity,OfNat.ofNat]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.Correct
