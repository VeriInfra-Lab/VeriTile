import bench.repairs.tritonbench_g.quantize_kv_transform.IO
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.FPEquiv
open VeriTile.Triton FP.Structural KernelSource KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- Same complete source token grid and two outputs, with the existing native
ordered reduction, full divisions and selected PTX signed-byte conversion. -/
abbrev quantizeIO (a : Arithmetic) (rest : Algebra Word) (p : Parameters)
    (count : Nat) (order : List Nat) :=
  IO.quantizeIO (KernelNativeProfile.backend a rest) p count order

/-- Complete source safety and execution, exact independent native codes and
scales, and the entire exterior frame. All raw FP32 words remain admitted. -/
theorem kernel_native (a : Arithmetic) (rest : Algebra Word) (p : Parameters)
    (width : 0 < p.width) (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    (quantizeIO a rest p count order).ImplementsOn (IO.Destinations p count)
      (NativeIO.mathematical a p headCover colCover count) :=
  NativeIO.implements a rest p width headCover colCover count order coverage

#stmtSurfaceSubset kernel_native ⊆
  [FP.Structural.Algebra, Word, instInhabitedWord, Arithmetic, Parameters, Parameters.columns, Parameters.heads,
   Parameters.height, Parameters.width, quantizeIO, ProgramIO.ImplementsOn,
   IO.Destinations, NativeIO.mathematical, FP.Value.zero, FP.Format.fp32]
-- Arithmetic is the explicit scalar instruction backend, not a transformation law.
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.FPEquiv
