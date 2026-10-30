import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.RawGrid
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeArithmetic

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.KernelResults.Real
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete readable Q/K/V source grid over raw regions and launch IDs. The
independent target composes the exact RMS/matmul/rotary mathematical tile at
every scheduled program, retaining all outputs and every exterior typed cell. -/
theorem kernel_real (fp8 : Bool) (p : Fin 3 → Arguments)
    (initial : State ℝ) (epsilon theta : HostScalar)
    (he : initial.regs .real [] "EPS"=some (fun _ => epsilon.value))
    (ht : initial.regs .real [] "THETA"=some (fun _ => theta.value))
    (bytes : Int8Memory initial) (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (order : List (Nat → Nat)) :
    ∃ final,RawGrid.execute mathArithmetic Bool.false fp8 p initial order initial.mem=some final ∧
      final=RawGrid.mathematical fp8 p initial epsilon theta order initial.mem ∧
      (∀ r o,RawGrid.Outside p order r o → final r o=initial.mem r o) := by
  refine ⟨RawGrid.reference mathArithmetic Bool.false fp8 p initial epsilon.value theta.value order initial.mem,
    RawGrid.complete mathArithmetic math_zeroLaws Bool.false fp8 p initial epsilon.value theta.value
      he ht even apart order initial.mem bytes,
    RawGrid.reference_mathematical fp8 p initial epsilon theta order initial.mem,?_⟩
  exact RawGrid.reference_frame mathArithmetic Bool.false fp8 p initial epsilon.value theta.value order initial.mem

#axiomsClean kernel_real
#specNonCircular RawGrid.mathematical avoiding [RawGrid.execute,run,qkvCode,sourceTileValues,nativeTileValues]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.KernelResults.Real

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.KernelResults.Native
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete selected native Q/K/V source grid and independent ordered raw-word
reference, including all narrow output cells and the complete exterior frame.
The existing primitive zero facts, K8 MMA port, reduction tree and ordered FMA
selection are preserved, with no generic source/native arithmetic equality. -/
theorem kernel_native (instructions : Native.Primitives)
    (zero : Native.ZeroRealization instructions) (fp8 : Bool) (p : Fin 3 → Arguments)
    (initial : State Native.Word) (epsilon theta : Native.Word)
    (he : initial.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : initial.regs .real [] "THETA"=some (fun _ => theta))
    (bytes : Int8Memory initial) (even : ∀ phase,(p phase).N%2=0)
    (apart : ∀ writer reader,(p reader).X≠(p writer).OUT ∧
      (p reader).W≠(p writer).OUT ∧ (p reader).RMS≠(p writer).OUT)
    (order : List (Nat → Nat)) :
    ∃ final,RawGrid.execute (Native.arithmetic instructions) Bool.true fp8 p initial order initial.mem=some final ∧
      final=RawGrid.reference (Native.arithmetic instructions) Bool.true fp8 p initial epsilon theta order initial.mem ∧
      (∀ r o,RawGrid.Outside p order r o → final r o=initial.mem r o) := by
  refine ⟨RawGrid.reference (Native.arithmetic instructions) Bool.true fp8 p initial epsilon theta order initial.mem,
    RawGrid.complete (Native.arithmetic instructions) (Native.zero_laws instructions zero)
      Bool.true fp8 p initial epsilon theta he ht even apart order initial.mem bytes,rfl,?_⟩
  exact RawGrid.reference_frame (Native.arithmetic instructions) Bool.true fp8 p initial epsilon theta order initial.mem

#axiomsClean kernel_native
#specNonCircular RawGrid.reference avoiding [RawGrid.execute,run,qkvCode,sourceTileValues,nativeTileValues]
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain.KernelResults.Native
