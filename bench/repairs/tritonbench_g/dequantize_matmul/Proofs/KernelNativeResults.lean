import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelWhole

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelNativeResults
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

/-- Both complete raw kernel grids terminate, produce the independent signed
INT8 dequantization formula, and preserve every cell outside the output span.
The input contract is raw shape/tile geometry, region separation and initial
signed-byte/scale reads, with no Tensor metadata or wrapper objects. -/
theorem kernel_equiv (multiply : Word → Word → Word) (rest : Algebra Word) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p)
    (q : Nat → BitVec 8) (scale : Nat → Word) (initial : State Word)
    (reads : Reads p q scale initial) :
    ∃ final, KernelGrid.run (nativeAlgebra multiply rest) p Bool.false (KernelGrid.order p) initial = some final ∧
      KernelGrid.run (nativeAlgebra multiply rest) p Bool.true (KernelGrid.order p) initial = some final ∧
      Result p (fun i => nativeWord multiply (q i) (scale (i % p.N))) initial final :=
  KernelWhole.native_correct multiply rest p hK hN sep q scale initial reads

alias unique_output_owner := KernelGrid.unique_owner
#axiomsClean kernel_equiv
#axiomsClean unique_output_owner
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelNativeResults
