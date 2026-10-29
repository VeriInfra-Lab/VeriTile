import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelCoverage
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelRealOutput
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNativeProfile
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.PublicRawTransport

/- Exact previous arbitrary-state endpoints, including the original all-offset
ByteMemory condition, raw input relation, complete launch and complete frame.
PublicRawTransport binds these states to the shared IO and identical schedule. -/

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.RawEndpoints
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent RMS projection and optional rotary transformation on raw inputs. -/
def kernelRealTarget (d8 : Float8E5Bits → ℝ) (fp8 : Bool) (p : KernelSource.Parameters)
    (v : KernelGraph.Input ℝ) (x : KernelGrid.Coordinate B p) : ℝ :=
  KernelMath.output p.K p.headDim p.start p.rotary
    (KernelReal.logical d8 fp8 v x.1.val x.2.1.val) x.2.1.val x.2.2.val

/-- Complete original-grid execution, all mathematical outputs and the full raw
memory frame. Shape, strides, storage format and input cells are kernel data. -/
theorem rms_rbe_matmul_grid_real
    (d8 : Float8E5Bits → ℝ) (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters) (evenN : p.N%2=0)
    (before : State ℝ) (bytes : KernelSource.ByteMemory before)
    (v : KernelGraph.Input ℝ) (reads : KernelSource.inputs p before=v) :
    ∃ final, KernelGrid.launch (KernelReal.operations d8) fp8 B p before=some final ∧
      (∀ x : KernelGrid.Coordinate B p, final.mem "OUT" (KernelGrid.flat p x)=
        Cell.mk (KernelSource.outputDType fp8).toTileDType
          (ofFloat (KernelSource.outputDType fp8) (kernelRealTarget d8 fp8 p v x))) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"OUT" ∨ ∀ x : KernelGrid.Coordinate B p,off≠KernelGrid.flat p x) →
        final.mem r off=before.mem r off) := by
  obtain ⟨final,run,out,frame⟩ := KernelGrid.logical_complete (KernelReal.operations d8) fp8 B p before bytes
  refine ⟨final,run,?_,frame⟩
  intro x
  have typed := out x
  unfold KernelGrid.result at typed
  rw [reads] at typed
  obtain ⟨hrow,hcol⟩ := KernelGrid.owner_coordinate p x
  change (KernelGrid.tile p (KernelGrid.owner p x)).row*16+(KernelGrid.lane x).1.val=x.2.1.val at hrow
  change (KernelGrid.tile p (KernelGrid.owner p x)).column*64+(KernelGrid.lane x).2.1.val=x.2.2.val at hcol
  have real := KernelReal.output_real d8 zero fp8 p v
    (KernelGrid.tile p (KernelGrid.owner p x)) (KernelGrid.lane x)
    (by rw [hrow]; exact x.2.1.isLt) (by rw [hcol]; exact x.2.2.isLt) evenN
  rw [real,hrow,hcol] at typed
  exact typed

#axiomsClean rms_rbe_matmul_grid_real
#specNonCircular kernelRealTarget avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGraph.output,KernelGrid.result,KernelGrid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.RawEndpoints

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.RawEndpoints
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open KernelNative (Term Instructions Profile Reductions Dots)
noncomputable section

def kernelReadOutput (fp8 : Bool) (s : State Term) (off : Nat) : Term :=
  if fp8 then (s.mem "OUT" off).read .f8e5 else (s.mem "OUT" off).read .fp16

/-- Complete source execution and every native output observation, under the
existing selected scalar/conversion/K8 instruction and compiler profile. -/
theorem rms_rbe_matmul_grid_native
    (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (fp8 : Bool) (B : Nat) (p : KernelSource.Parameters)
    (before : State Term) (bytes : KernelSource.ByteMemory before)
    (v : KernelGraph.Input Term) (reads : KernelSource.inputs p before=v) :
    ∃ final, KernelGrid.launch (KernelNative.operations R D) fp8 B p before=some final ∧
      (∀ x : KernelGrid.Coordinate B p,
        KernelNative.eval I P.schedule (kernelReadOutput fp8 final (KernelGrid.flat p x))=
          KernelNative.reference I P.schedule R D fp8 p v
            (KernelGrid.tile p (KernelGrid.owner p x)) (KernelGrid.lane x)) ∧
      (∀ (r : RegionName) (off : Nat),
        (r≠"OUT" ∨ ∀ x : KernelGrid.Coordinate B p,off≠KernelGrid.flat p x) →
        final.mem r off=before.mem r off) := by
  obtain ⟨final,run,out,frame⟩ := KernelGrid.logical_complete (KernelNative.operations R D) fp8 B p before bytes
  refine ⟨final,run,?_,frame⟩
  intro x
  have typed := out x
  unfold KernelGrid.result at typed
  rw [reads] at typed
  unfold kernelReadOutput
  rw [typed]
  cases fp8 <;> simp [KernelSource.outputDType,ofFloat,Cell.read,KernelNative.reference]

#axiomsClean rms_rbe_matmul_grid_native
#specNonCircular KernelNative.reference avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGrid.result,KernelGrid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.RawEndpoints
