import bench.repairs.tritonbench_g.chunked_cumsum_fwd.KernelOnly.IO
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.PairSemantics
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairIO
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math PairSource PairSemantics
noncomputable section

def originalIO (cfg : FP.Config) (M : NativeModel (FP.Value .fp32)) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : FP.Value .fp32) :=
  Interface.programIO (concreteModel cfg M) c plan hasBias useSoftplus lower upper

/-- The actual scale-first quotation uses the same natural graph interpreter,
runtime bound bindings, launch order, channels, masks, encodings and observer. -/
def candidateInvocation (M : NativeModel α) (c : Geometry) (plan : PrefixPlan c.BH c.BC)
    (hasBias useSoftplus : Bool) (lower upper : α) (l : Launch) : Invocation (Term α) :=
  { Interface.invocation M c plan hasBias useSoftplus lower upper l with
      kernel := candidateKernel c hasBias useSoftplus }

def scaleFirstIO (cfg : FP.Config) (M : NativeModel (FP.Value .fp32)) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : FP.Value .fp32) :
    ProgramIO.Variant (originalIO cfg M c plan hasBias useSoftplus lower upper) where
  schedule := fun _ => c.grid.map
    (candidateInvocation (concreteModel cfg M) c plan hasBias useSoftplus lower upper)
  observe := (originalIO cfg M c plan hasBias useSoftplus lower upper).observe

def original (cfg : FP.Config) (M : NativeModel (FP.Value .fp32)) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : FP.Value .fp32) :=
  ProgramIO.Implementation.original
    (originalIO cfg M c plan hasBias useSoftplus lower upper) (fun _ => True)

def scaleFirst (cfg : FP.Config) (M : NativeModel (FP.Value .fp32)) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : FP.Value .fp32) :=
  ProgramIO.Implementation.ofVariant
    (scaleFirstIO cfg M c plan hasBias useSoftplus lower upper) (fun _ => True)

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.PairIO
