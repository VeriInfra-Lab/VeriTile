import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelScalar

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelNativeResults
open VeriTile.Triton FP.Structural KernelScalar Storage
noncomputable section
/-- Complete forward source grid over raw dimensions, strides, addresses and initial memory; independent output target and full memory frame. -/
theorem forward (p : Parameters) (b : Physical.Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ final,Grid.launch (forwardBackend selectedProfile) .forward p (Physical.backed .forward p b heap)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 11,final.mem outputRegion (r*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (Packing.word (forwardReference selectedProfile b heap r c))) ∧
      Frame .forward p (Physical.backed .forward p b heap) final :=
  KernelScalar.native_forward selectedProfile p b heap fit

/-- Complete backward source grid over raw dimensions, strides, addresses and initial memory; independent output target and full memory frame. -/
theorem backward (p : Parameters) (b : Physical.Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ final,Grid.launch (backwardBackend selectedProfile) .backward p (Physical.backed .backward p b heap)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 3,final.mem gradientRegion (r*3+c.val)=
        .mk .real (Packing.word (backwardReference selectedProfile p b heap r c))) ∧
      Frame .backward p (Physical.backed .backward p b heap) final :=
  KernelScalar.native_backward selectedProfile p b heap fit

#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelNativeResults
