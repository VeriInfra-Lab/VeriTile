import bench.tritonbench_g.ksoftmax_triton.Sources.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
open Lean VeriTile.Triton

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat →
  Bool → Option String → Bool → Bool → ComputeKernel

/-- The actual selected forward source, with the IO ports required by the
existing forward interface. Its body is the shared source quotation. -/
def fp32Original : Source := fun Y X M ym yn xm xn sm K D log mode causal fp16 =>
  ComputeKernel.mk [X,M] [Y]
    (VeriTile.Bench.Repairs.Ksoftmax.forward Y X M ym yn xm xn sm K D log mode causal fp16).surfaceBody

theorem source_body (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log causal fp16 : Bool) (mode : Option String) :
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).surfaceBody =
      (VeriTile.Bench.Repairs.Ksoftmax.forward Y X M ym yn xm xn sm K D log mode causal fp16).surfaceBody := rfl

theorem fp32_projects (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log causal fp16 : Bool) (mode : Option String) :
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).toAlgorithm? =
      .ok (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).toAlgKernel := by
  rfl

theorem io_signature (Y X M : RegionName) (ym yn xm xn sm K D : Nat)
    (log causal fp16 : Bool) (mode : Option String) :
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).inputs = [X, M] ∧
    (fp32Original Y X M ym yn xm xn sm K D log mode causal fp16).outputs = [Y] := by
  exact ⟨rfl, rfl⟩

#axiomsClean fp32_projects
#axiomsClean io_signature
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.Typed
