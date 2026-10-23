import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Defect
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Real
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Lowering

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics
open VeriTile.Triton FP.Structural Host
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 64000

/-- All returned selected channels equal the original exact-decimal forward
polynomials, on the separately interpreted arbitrary Real coordinate tensor.
The same complete source launch preserves every unselected logical cell. -/
theorem correct_forward (rest : Algebra ℝ) (q : Request) (admit : Admitted q)
    (allocated : TensorObject) (allocation : Allocation q allocated) (memory : RealProof.Heap) :
    let p:=parameters q admit allocated
    let before:=RealProof.initial q.coordinates.metadata.pointer 0 (q.output allocated).metadata.pointer memory Bool.false
    ∃ final,Grid.launch (ScalarReal.backend rest) .forward p before=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 11,
        final.mem outputRegion (r*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (Polynomial.value c
          (memory (q.coordinates.metadata.pointer+4*(r*3)))
          (memory (q.coordinates.metadata.pointer+4*(r*3+1)))
          (memory (q.coordinates.metadata.pointer+4*(r*3+2))))) ∧
      (∀ region index,(region≠outputRegion ∨ ∀ r,r<p.rows → ∀ c : Fin 11,
        index≠r*p.outputWidth+p.columnOffset+c.val) → final.mem region index=before.mem region index) := by
  exact RealProof.forward rest (parameters q admit allocated) (parameters_fit q admit allocated allocation)
    q.coordinates.metadata.pointer (q.output allocated).metadata.pointer memory

/-- Every returned coordinate gradient is the true Jacobian-transpose product
of the eleven original decimal polynomials. This covers the complete source
backward launch, including the retained load-zero-and-add accumulation. -/
theorem correct_backward (rest : Algebra ℝ) (q : Request) (admit : Admitted q)
    (allocated : TensorObject) (allocation : Allocation q allocated)
    (upstream gradient : TensorObject)
    (upValid : UpstreamContract q.coordinates.metadata upstream.metadata (q.output allocated).metadata.columns)
    (fresh : FreshGradient q.coordinates.metadata upstream.metadata gradient)
    (memory : RealProof.Heap) :
    let p:=backwardParameters (parameters q admit allocated) upstream.metadata
    let before:=RealProof.initial q.coordinates.metadata.pointer upstream.metadata.pointer gradient.metadata.pointer memory Bool.true
    ∃ final,Grid.launch (ScalarReal.backend rest) .backward p before=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 3,final.mem gradientRegion (r*3+c.val)=
        .mk .real (Polynomial.vjp
          (memory (q.coordinates.metadata.pointer+4*(r*3)))
          (memory (q.coordinates.metadata.pointer+4*(r*3+1)))
          (memory (q.coordinates.metadata.pointer+4*(r*3+2)))
          (fun j=>memory (upstream.metadata.pointer+4*(r*upstream.metadata.rowStride.toNat+
            (q.column.toNat+j.val)*upstream.metadata.colStride.toNat))) c)) ∧
      (∀ region index,(region≠gradientRegion ∨ ∀ r,r<p.rows → ∀ c : Fin 3,
        index≠r*3+c.val) → final.mem region index=before.mem region index) := by
  exact RealProof.backward rest (backwardParameters (parameters q admit allocated) upstream.metadata)
    (parameters_fit q admit allocated allocation) q.coordinates.metadata.pointer upstream.metadata.pointer gradient.metadata.pointer memory

/-- The actual Python guard succeeds exactly on the stated admitted domain. -/
theorem checked_forward_domain (x : TensorArgument) (y : Option TensorArgument)
    (block column : IntegerArgument) :
    forwardCheck x y block column=.ok () ↔ AdmittedForward x y block column := forwardCheck_iff x y block column

theorem checked_backward_domain (x : TensorView) (upstream : TensorArgument) (width : Nat) :
    backwardCheck x upstream width=.ok () ↔ AdmittedBackward x upstream width := backwardCheck_iff x upstream width

/-- These derivatives were proved independently of the repaired backward graph. -/
theorem exact_partial_derivatives (channel : Fin 11) (component : Fin 3) (x y z : ℝ) :
    HasDerivAt (fun t=>Polynomial.value channel
      (if component=0 then t else x) (if component=1 then t else y) (if component=2 then t else z))
      (Polynomial.jacobian channel component x y z) (Polynomial.coordinate component x y z) :=
  Polynomial.exact_partial_derivative channel component x y z

#axiomsClean correct_forward
#axiomsClean correct_backward
#axiomsClean checked_forward_domain
#axiomsClean checked_backward_domain
#axiomsClean exact_partial_derivatives
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics
