import bench.tritonbench_g.flash_decode2_phi.IO
import bench.tritonbench_g.flash_decode2_phi.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Correct
open VeriTile.Triton FP.Structural FlashDecode2
open scoped BigOperators
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Exact complete masked Phi source with the IO channels, dynamic masks,
strides, FP32 interpretation and output windows listed in IO.lean. Finite
summary operands are encoded as some Real; negative infinity stays internal. -/
abbrev decodeIO (g : PhiIO.Geometry) (order : List Nat) :=
  LogicalIO.view g order RealIO.observe

/-- Independent exponentially weighted summary at every output coordinate.
Fits describes backing coverage, separate from source stride S; Positive is
exactly the original nonempty-summary premise. Both empty grids remain valid. -/
specification kernel_real (g : PhiIO.Geometry) (width : g.D≤g.BD) (order : List Nat)
    (coverage : order.Perm (List.range (g.B*g.H))) :
    Spec.Real ((decodeIO g order) ⊨{fun xs => LogicalIO.Fits g xs ∧ LogicalIO.Positive g xs}
      (fun xs row dim =>
        (∑ j ∈ Finset.range (xs.steps row), Real.exp (xs.log row j) * xs.mid row j dim) /
          (∑ j ∈ Finset.range (xs.steps row), Real.exp (xs.log row j)))) :=
  (LogicalIO.judgment_iff g order).mpr (RealIO.implements g width order coverage)

#stmtSurfaceSubset kernel_real ⊆
  [LogicalIO.Inputs,Spec.Real,PhiIO.Geometry,PhiIO.Geometry.B,PhiIO.Geometry.H,PhiIO.Geometry.D,
   PhiIO.Geometry.BD,LogicalIO.Fits,LogicalIO.Positive,LogicalIO.Inputs.steps,
   decodeIO,LogicalProgramIO.ImplementsOn,LogicalIO.Inputs.length,LogicalIO.Inputs.log,LogicalIO.Inputs.mid]
#axiomsClean kernel_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.Correct
