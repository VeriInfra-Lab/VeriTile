import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.PublicForwardIO
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.PublicChunkedIO
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.PublicBackwardIO
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelCompiler

namespace VeriTile.Bench.Repairs.FastCeLoss.NativeIO
open VeriTile.Triton FP.Structural KernelSource PublicIO
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
noncomputable section

def forwardFormula (R : Reductions) (I : Instructions) (S : Schedule) (p : Parameters)
    (xs : ChannelValues Term (forwardInputs p Bool.false)) : ChannelValues Word (forwardOutputs p) :=
  fun o j => mapValue (KernelCompiler.observe I S) (forwardOutputs p)[o].dtype
    (PublicIO.forwardFormula (algebra R) p Bool.false xs o j)
def chunkedFormula (R : Reductions) (I : Instructions) (S : Schedule) (p : Parameters) (chunk : Nat)
    (xs : ChannelValues Term (forwardInputs p Bool.false)) : ChannelValues Word (forwardOutputs p) :=
  fun o j => mapValue (KernelCompiler.observe I S) (forwardOutputs p)[o].dtype
    (PublicIO.chunkedFormula (algebra R) p chunk Bool.false xs o j)
def backwardFormula (R : Reductions) (I : Instructions) (S : Schedule) (p : Parameters) (chunk : Nat)
    (xs : ChannelValues Term (backwardInputs p Bool.false)) : ChannelValues Word (backwardOutputs p) :=
  fun o j => mapValue (KernelCompiler.observe I S) (backwardOutputs p)[o].dtype
    (PublicIO.backwardFormula (algebra R) p chunk Bool.false xs o j)

theorem forward_observes (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (row : Nat) (positive : 0<p.block) (distinct : p.loss≠p.lse) :
    (forwardIO (algebra R) id id p row 0 Bool.false Bool.false).ObservesOn
      (fun _ _ => KernelCompiler.observe I S) (nativeDomain p Bool.false)
      (forwardFormula R I S p) :=
  ProgramIO.ObservesOn.of_implementsOn
    (forward_implements (algebra R) p row Bool.false positive distinct) _

theorem chunked_observes (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (row chunk : Nat) (positive : 0<p.block) (distinct : p.loss≠p.lse) :
    (forwardIO (algebra R) id id p row chunk Bool.true Bool.false).ObservesOn
      (fun _ _ => KernelCompiler.observe I S) (nativeDomain p Bool.false)
      (chunkedFormula R I S p chunk) :=
  ProgramIO.ObservesOn.of_implementsOn
    (chunked_implements (algebra R) p row chunk Bool.false positive distinct) _

theorem backward_observes (R : Reductions) (I : Instructions) (S : Schedule)
    (p : Parameters) (row chunk : Nat) :
    (backwardIO (algebra R) id id p row chunk Bool.false).Observes
      (fun _ _ => KernelCompiler.observe I S) (backwardFormula R I S p chunk) :=
  ProgramIO.Observes.of_implements (backward_implements (algebra R) p row chunk Bool.false) _

#specNonCircular forwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular chunkedFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular backwardFormula avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward,FP.Structural.exec,FP.Structural.run]
#axiomsClean forward_observes
#axiomsClean chunked_observes
#axiomsClean backward_observes
end
end VeriTile.Bench.Repairs.FastCeLoss.NativeIO
