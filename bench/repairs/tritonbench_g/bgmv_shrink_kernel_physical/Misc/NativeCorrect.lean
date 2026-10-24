import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PublicIO

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry records the remaining comparison blocker. -/

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPEquiv
open VeriTile.Triton FP.Structural NativeGraph
open VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset
noncomputable section

/-- The unchanged complete source retains half/bfloat16 multiply and reduce,
FP32 accumulation/scaling, output conversion and each address's atomic order.
Inputs and the runtime scaling argument remain arbitrary native terms. -/
abbrev io (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (h : Call) (indices : Nat → Int)
    (scale : NG.Term) (orders : Orders h) :=
  IO.programIO algebra (primitives plans) h indices scale orders
    (fun term => (binding.program term).eval I)

/-- Source-to-native observation under the existing local compiler/scalar
model. This is no arithmetic rewrite and assumes no equality with Real. -/
theorem kernel_native (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (h : Call) (indices : Nat → Int)
    (scale : NG.Term) (orders : Orders h) :
    (io plans S I binding h indices scale orders).ImplementsOnCells
      (IO.domain h indices) (Mathematical.native plans S I h indices scale orders) :=
  PublicIO.native plans S I binding h indices scale orders

#stmtSurfaceSubset kernel_native ⊆
  [ScalarTree.Plans, Selection, Instructions, CompilerBinding, Call, NG.Term, NG.instInhabitedTerm, Orders,
    VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Geometry.Program,
    io, PerAddressProgramIO.ImplementsOnCells, IO.domain, Mathematical.native]
#specNonCircular Mathematical.native avoiding [Surface.source, FP.Structural.exec, FPValues.target]
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPEquiv
