import bench.tritonbench_g.flash_decode2_llama.IO
import bench.tritonbench_g.flash_decode2_llama.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.Correct
open VeriTile.Triton FP.Structural FlashDecode2
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- IO.lean exposes the complete unmasked Llama source schedule, FP32 E/M/O
channels, natural L metadata, dense windows and all output rows. The ideal
interpreter retains negative infinity as Option Real; outputs are proved some. -/
abbrev decodeIO (g : KernelInterface.Geometry) (order : List Nat) :=
  LogicalIO.view g order RealIO.observe

/-- Actual full-grid source safety and execution, independent weighted summary
at every coordinate, and complete exterior memory frame. -/
specification kernel_real (g : KernelInterface.Geometry) (order : List Nat)
    (coverage : order.Perm (List.range (g.B * g.H))) :
    Spec.Real ((decodeIO g order) ⊨{LogicalIO.Lengths g} (fun xs row dim =>
        let count := (xs.length row+g.Q-1)/g.Q
        (∑ j ∈ Finset.range count, Real.exp (xs.log row j) * xs.mid row j dim) /
          (∑ j ∈ Finset.range count, Real.exp (xs.log row j)))) :=
  (LogicalIO.judgment_iff g order).mpr (RealIO.implements g order coverage)

#stmtSurfaceSubset kernel_real ⊆
  [LogicalIO.Inputs,Spec.Real,KernelInterface.Geometry,KernelInterface.Geometry.B,KernelInterface.Geometry.H,KernelInterface.Geometry.D,
   KernelInterface.Geometry.Q,LogicalIO.Lengths,
   decodeIO,LogicalProgramIO.ImplementsOn,LogicalIO.Inputs.length,LogicalIO.Inputs.log,LogicalIO.Inputs.mid]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.Correct
