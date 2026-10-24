import bench.repairs.tritonbench_g.geglu_tanh_triton.Misc.NativeIO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Proofs.PublicIO
import bench.repairs.tritonbench_g.geglu_tanh_triton.Misc.RawNativeApplication

/-! Auxiliary forward/backward native word observations. The source index-commutation equivalence is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.GegluTanhTriton
open VeriTile.Triton FP.Structural FP.ProfiledMemory Native
open scoped VeriTile.Spec
noncomputable section

/-- Independent native word observation for either complete source grid.
All FP32 words, in-place backward outputs, ternary FMA, libdevice tanh branches,
FTZ sites and source constants retain their existing primitive semantics. -/
theorem geglu_kernel_fp32 (instructions : Instructions) (candidate backward : Bool)
    (count : Nat) (order : List Nat) (legal : order.Perm (List.range (Grid.grid count))) :
    (Reference.nativeIO instructions candidate backward count order).Implements
      (Reference.native instructions candidate backward count order) :=
  PublicIO.native instructions candidate backward count order legal

#specNonCircular Reference.native avoiding [source,indexCommutedSource,selectedSource,Tile.scalar]

#stmtSurfaceSubset geglu_kernel_fp32 ⊆
  [ProgramIO.Implements,Reference.nativeIO,Reference.native,Grid.grid,Instructions,Native.Word,
   VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.Term,
   VeriTile.Bench.TritonBenchG.SwigluBackward.Observed.instInhabitedTerm]

#axiomsClean geglu_kernel_fp32

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton
