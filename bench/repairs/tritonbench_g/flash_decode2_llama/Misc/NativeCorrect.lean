/- Auxiliary single-source native correctness. No program-pair equivalence is claimed. -/
import bench.repairs.tritonbench_g.flash_decode2_llama.IO
import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.FPEquiv
open VeriTile.Triton FP.Structural FlashDecode2
noncomputable section

/-- Same actual complete source and channels as Real, interpreted by the
existing native term backend and fixed FMA/operand-order observer. All input
words are retained; there is no finite-only domain or floating rewrite law. -/
abbrev decodeIO (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat) :
    ProgramIO Native.Word Native.Term :=
  IO.decodeIO Native.algebra g order Native.Term.bits (NativeIO.observe p g)

/-- Full source access safety, successful complete grid, independent native
instruction summary and the exact exterior frame under the primitive model. -/
theorem kernel_native (p : RawNative.Ops) (g : KernelInterface.Geometry)
    (order : List Nat) (coverage : order.Perm (List.range (g.B * g.H))) :
    (decodeIO p g order).ImplementsOn (IO.Lengths g) (NativeIO.mathematical p g) :=
  NativeIO.implements p g order coverage

#stmtSurfaceSubset kernel_native ⊆
  [RawNative.Ops, Native.Word, Native.Term, Native.instInhabitedTerm, KernelInterface.Geometry, KernelInterface.Geometry.B, KernelInterface.Geometry.H,
   decodeIO, ProgramIO.ImplementsOn, IO.Lengths, NativeIO.mathematical]
-- RawNative.Ops is the primitive instruction interpretation, not a rewrite assumption.
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.FPEquiv
