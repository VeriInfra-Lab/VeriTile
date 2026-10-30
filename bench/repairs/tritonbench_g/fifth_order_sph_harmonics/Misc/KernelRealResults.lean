import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelScalar
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Defect

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelRealResults
open VeriTile.Triton FP.Structural KernelScalar Storage
noncomputable section
/-- Complete forward source grid over raw dimensions, strides, addresses and initial memory; independent output target and full memory frame. -/
theorem forward (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ final,Grid.launch realBackend .forward p (realInitial p b memory)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 11,final.mem outputRegion (r*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (realForward b memory r c)) ∧ Frame .forward p (realInitial p b memory) final :=
  KernelScalar.real_forward p b memory fit

/-- Complete backward source grid over raw dimensions, strides, addresses and initial memory; independent output target and full memory frame. -/
theorem backward (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ)
    (fit : p.columnOffset+11≤p.outputWidth)
    (zero : ∀ r,r<p.rows → ∀ c : Fin 3,memory (b.destination+4*(r*3+c.val))=0) :
    ∃ final,Grid.launch realBackend .backward p (realInitial p b memory)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 3,final.mem gradientRegion (r*3+c.val)=
        .mk .real (realBackward p b memory r c)) ∧ Frame .backward p (realInitial p b memory) final :=
  KernelScalar.real_backward p b memory fit zero

/-- Independent monomial-table differentiation covers all 33 partials. -/
theorem exact_partial_derivatives (channel : Fin 11) (component : Fin 3) (x y z : ℝ) :
    HasDerivAt (fun t=>Polynomial.value channel
      (if component=0 then t else x) (if component=1 then t else y) (if component=2 then t else z))
      (Polynomial.jacobian channel component x y z) (Polynomial.coordinate component x y z) :=
  Polynomial.exact_partial_derivative channel component x y z

#axiomsClean forward
#axiomsClean backward
#axiomsClean exact_partial_derivatives
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelRealResults
