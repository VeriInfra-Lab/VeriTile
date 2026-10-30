import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelCoverage
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNativeProfile

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelNativeComplete
open VeriTile.Triton FP.Structural
open Contract (Inputs InputContract)
open KernelBinding (Bytes Cast Preparation parameters fp8)
open KernelNative (Term Instructions Profile Reductions Dots)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

abbrev Coordinate (i : Inputs) := KernelGrid.Coordinate (Checked.shape i).B (parameters i)
def expression (R : Reductions) (D : Dots) (i : Inputs) (a : Checked.Allocation i)
    (cast : Cast) (before : Bytes) (x : Coordinate i) : Term :=
  KernelGraph.output (KernelNative.operations R D) (fp8 i) (parameters i)
    (KernelBinding.originalInputs i a cast before)
    (KernelGrid.tile (parameters i) (KernelGrid.owner (parameters i) x)) (KernelGrid.lane x)
def reference (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (i : Inputs) (a : Checked.Allocation i) (cast : Cast) (before : Bytes) (x : Coordinate i) :=
  KernelNative.eval I P.schedule (expression R D i a cast before x)

theorem flat_bound (i : Inputs) (a : Checked.Allocation i) (x : Coordinate i) :
    KernelGrid.flat (parameters i) x<a.out.metadata.numel := by
  rw [a.outCount]
  simpa [KernelGrid.flat,parameters,RmsMatmulRbe.Checked.flat3,Nat.add_mul,Nat.mul_assoc] using
    RmsMatmulRbe.Checked.flat3_lt _ _ _ _ _ _ x.1.isLt x.2.1.isLt x.2.2.isLt

/-- Actual original-grid execution succeeds from the physical argument reads.
The result expression contains only original inputs and their scalar RMS
conversion. No native output or execution premise occurs in the hypotheses. -/
theorem terms_complete (R : Reductions) (D : Dots) (i : Inputs) (h : InputContract i)
    (a : Checked.Allocation i) (cast : Cast) (before prepared : Bytes)
    (prep : KernelBinding.Preparation i a cast before prepared) (scratch : State Term) :
    ∃ final, KernelGrid.launch (KernelNative.operations R D) (fp8 i) (Checked.shape i).B (parameters i)
      (KernelBinding.state i a prepared scratch)=some final ∧
      (∀ x : Coordinate i, KernelPhysical.readOutput i final (KernelGrid.flat (parameters i) x)=
        expression R D i a cast before x) ∧
      (∀ (r : RegionName) (off : Nat), (r≠"OUT" ∨ ∀ x : Coordinate i,off≠KernelGrid.flat (parameters i) x) →
        final.mem r off=(KernelBinding.state i a prepared scratch).mem r off) := by
  obtain ⟨final,run,out,frame⟩ := KernelGrid.logical_complete (KernelNative.operations R D) (fp8 i)
    (Checked.shape i).B (parameters i) (KernelBinding.state i a prepared scratch)
    (KernelBinding.state_bytes i a prepared scratch)
  refine ⟨final,run,?_,frame⟩
  intro x
  have typed := out x
  unfold KernelGrid.result at typed
  rw [KernelBinding.state_inputs] at typed
  have read : KernelPhysical.readOutput i final (KernelGrid.flat (parameters i) x)=
      KernelGraph.output (KernelNative.operations R D) (fp8 i) (parameters i)
        (KernelBinding.preparedInputs i a prepared)
        (KernelGrid.tile (parameters i) (KernelGrid.owner (parameters i) x)) (KernelGrid.lane x) := by
    unfold KernelPhysical.readOutput
    rw [typed]
    cases hf : fp8 i <;> simp [hf,KernelSource.outputDType,ofFloat,Cell.read]
  rw [read]
  apply KernelGraph.output_congr
  · have := x.2.1.isLt; omega
  · have := x.2.2.isLt; omega
  · exact KernelBinding.original_agreement i h a cast before prepared prep x.1.val x.1.isLt

/-- Returned physical bytes, complete exterior frame, and every byte of every
original input backing. The only deferred numerical ports are individual
instructions/conversions and fixed K8 MMA lanes in `Instructions`. -/
theorem complete (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (i : Inputs) (h : InputContract i) (a : Checked.Allocation i) (returns : Checked.Returns i a)
    (cast : Cast) (before prepared : Bytes) (prep : KernelBinding.Preparation i a cast before prepared)
    (scratch : State Term) :
    ∃ final, KernelGrid.launch (KernelNative.operations R D) (fp8 i) (Checked.shape i).B (parameters i)
      (KernelBinding.state i a prepared scratch)=some final ∧
      (∀ x : Coordinate i, ∀ k : Fin (KernelPhysical.storeWidth i),
        KernelPhysical.outputBytes I P.schedule i a prepared final
          ((RmsMatmulRbe.Checked.pointer64 returns.tensor (KernelGrid.flat (parameters i) x)).toNat+k.val)=
          KernelPhysical.byteOf (reference I P R D i a cast before x) k.val) ∧
      (∀ byte, KernelPhysical.Outside i a byte →
        KernelPhysical.outputBytes I P.schedule i a prepared final byte=prepared byte) ∧
      (∀ t, t∈i.reads → ∀ byte, byte<t.bytes →
        KernelPhysical.outputBytes I P.schedule i a prepared final (t.base+byte)=before (t.base+byte)) := by
  obtain ⟨final,run,out,_⟩ := terms_complete R D i h a cast before prepared prep scratch
  refine ⟨final,run,?_,KernelPhysical.frame I P.schedule i a prepared final,?_⟩
  · intro x k
    have hp : RmsMatmulRbe.Checked.pointer64 returns.tensor (KernelGrid.flat (parameters i) x)=
        RmsMatmulRbe.Checked.pointer64 a.out (KernelGrid.flat (parameters i) x) := by
      simp only [RmsMatmulRbe.Checked.pointer64,returns.view.pointer,returns.view.dtype]
    rw [hp,KernelPhysical.pointer_exact i h a ⟨_,flat_bound i a x⟩,
      KernelPhysical.coordinate_bytes I P.schedule i a prepared final ⟨_,flat_bound i a x⟩ k,out x]
    rfl
  · intro t ht byte hb
    rw [KernelPhysical.original_input_frame I P.schedule i h a prepared final t ht byte hb]
    exact prep.originalBytes t ht byte hb

#axiomsClean terms_complete
#axiomsClean complete
#specNonCircular reference avoiding [FP.Structural.exec,KernelSource.inputs,KernelSource.result,
  KernelSource.scatter,KernelGrid.result,KernelGrid.launch]
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelNativeComplete
