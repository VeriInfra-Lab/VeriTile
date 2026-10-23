import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.IOGrid

/-! The existing candidate changes only the exact integer N-mask product.
Signed INT8 promotion and source-ordered FP32 multiplication are unchanged.
The multiply implementation is the existing primitive model boundary; this
index transformation has no floating rewrite assumption. -/
namespace VeriTile.Bench.Repairs.DequantizeMatmul.NativeCorrect
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

/-- Both complete source grids use the public channels and windows in IO.lean. -/
def original (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters) :
    ProgramIO Word Word := PublicIO.sourceIO (nativeAlgebra multiply rest) p Bool.false

def indexCommuted (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters) :
    ProgramIO.Variant (original multiply rest p) := PublicIO.variant (nativeAlgebra multiply rest) p

/-- Both actual full grids safely execute, agree on every FP32 output word,
and preserve the exact exterior frame, for every signed INT8 input word. -/
theorem kernel_equiv (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p) :
    (original multiply rest p).EquivalentOn (indexCommuted multiply rest p) PublicIO.SignedBytes := by
  intro values _
  exact PublicIO.equivalent (nativeAlgebra multiply rest) p hK hN sep values trivial

#axiomsClean kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeMatmul.NativeCorrect
