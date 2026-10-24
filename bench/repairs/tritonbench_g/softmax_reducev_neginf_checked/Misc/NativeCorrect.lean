import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IONative
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.KernelNativeResults

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPEquiv
open VeriTile Triton FP.Structural
noncomputable section

/-- Complete source, typed metadata and all FP32 word inputs. Each actual
profile retains its reduction trees, replicas, clamp, FMA and final division. -/
def nativeIO (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :
    ProgramIO NativeGraph.Word NativeGraph.Node := PublicIO.nativeIO profiles ops c counts work

/-- Source-to-native word observation on every row. Instructions are the
existing seven primitive model families, not transformation assumptions;
no second kernel or new numerical equality is asserted. -/
theorem kernel_native (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job)
    (complete : work.Perm (Host.workset c)) :
    (nativeIO profiles ops c counts work).ImplementsOn PublicIO.LengthsMatch
      (PublicIO.nativeFormula profiles ops c counts) :=
  PublicIO.native_implements profiles ops c counts work complete

#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [ProgramIO.ImplementsOn,nativeIO,PublicIO.LengthsMatch,PublicIO.nativeFormula,
   Host.Call,Host.Call.batch,Host.Job,Host.workset,List.Perm,NativeGraph.Profile,
   NativeGraph.Instructions,NativeGraph.Word,NativeGraph.Node,NativeGraph.Expr,NativeGraph.instInhabitedExpr]
#specNonCircular PublicIO.nativeFormula avoiding
  [FPValue.source,FPValue.outputValues,FPValue.finalAccumulator,RawKernel.runGrid,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPEquiv
