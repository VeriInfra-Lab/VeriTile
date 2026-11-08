import bench.tritonbench_g.triton_linear_activation.Proofs.Support.Operations
import bench.tritonbench_g.triton_linear_activation.Proofs.Support.SourceKernels
namespace VeriTile.Bench.Repairs.TritonLinearActivation
variable {α : Type} {r : Request} {caller : Set Nat} {before : Memory α}
/-- Python materialization of the existing conditional width-index candidate. -/
def candidateSHA256 : String := "da9ace5661bbaaca884a1f736390465f733b68baf9ff0097a063ced50639629b"
/-- Host scheduling adapter used by the separate host results. The actual JIT
source used by the kernel specifications is bound below. -/
noncomputable def execute (h : Allocation r caller before) (a : Arithmetic α)
    (c : Fin 98) (currentDevice : Nat) (trials : List (Fin 98)) : HostResult α :=
  { h.hostResult a c currentDevice with memory := ((h.autotunedCode trials c).eval a h.initial).2 }

/-- Canonical source selected by the kernel revision. -/
def kernelSHA256 : String := "271cfcf8614afc2747837dbf8902f3addc9a3a605a6c1a85063ae2b6c346f8a9"

/-- Complete readable typed kernel, in actual source order. -/
noncomputable def kernel := KernelSource.source

end VeriTile.Bench.Repairs.TritonLinearActivation
