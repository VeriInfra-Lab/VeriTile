import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Proofs.Launch
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelRealComplete
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelTensorMetadata
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNativeComplete
import bench.repairs.tritonbench_g.rms_rbe_matmul_launch_guard.Misc.RawEndpoints

/-! Complete retained Tensor applications and input-check correspondence. -/
namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.TensorBridge
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Contract (Inputs InputContract)
open KernelBinding (Bytes Cast Preparation parameters fp8)
open KernelNative (Word)
noncomputable section

/-- Whole returned Tensor under exact-Real arithmetic, derived from actual
source execution on the original rectangular grid. The mathematical target
reads only original input words and their scalar RMS conversion. -/
theorem rms_rbe_matmul_repaired_real_returns
    (d32 d16 : Word → ℝ) (d8 : Float8E5Bits → ℝ) (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (i : Inputs) (h : InputContract i) (a : Checked.Allocation i) (returns : Checked.Returns i a)
    (cast : Cast) (before prepared : Bytes) (prep : KernelBinding.Preparation i a cast before prepared)
    (scratch : State ℝ) :
    Contract.LaunchAdmitted i ∧
    returns.tensor.metadata.shape=
      [(Checked.shape i).B,(Checked.shape i).M,(Checked.shape i).heads,(Checked.shape i).headDim] ∧
    returns.tensor.metadata.dtype=i.weight.metadata.dtype ∧
    returns.tensor.metadata.device=i.x.metadata.device ∧
    returns.tensor.metadata.dataPtr=a.out.metadata.dataPtr ∧
    ∃ final, KernelGrid.launch (KernelReal.operations d8) (fp8 i) (Checked.shape i).B (parameters i)
      (KernelRealBinding.state d32 d16 i a prepared scratch)=some final ∧
      (∀ x : KernelRealComplete.Coordinate i, final.mem "OUT" (KernelGrid.flat (parameters i) x)=
        Cell.mk (KernelSource.outputDType (fp8 i)).toTileDType
          (ofFloat (KernelSource.outputDType (fp8 i)) (KernelRealComplete.target d32 d16 d8 i a cast before x))) ∧
      (∀ (r : RegionName) (off : Nat), (r≠"OUT" ∨ ∀ x : KernelRealComplete.Coordinate i,off≠KernelGrid.flat (parameters i) x) →
        final.mem r off=(KernelRealBinding.state d32 d16 i a prepared scratch).mem r off) := by
  obtain ⟨hs,hd,hdev,hptr,_⟩ := KernelTensorMetadata.returned i a returns
  exact ⟨h.2,hs,hd,hdev,hptr,KernelRealComplete.complete d32 d16 d8 zero i h.1 a cast before prepared prep scratch⟩

#axiomsClean rms_rbe_matmul_repaired_real_returns
#specNonCircular KernelRealComplete.target avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGraph.output,KernelGrid.result,KernelGrid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.TensorBridge

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.TensorBridge
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Contract (Inputs InputContract)
open KernelBinding (Bytes Cast Preparation parameters fp8)
open KernelNative (Term Instructions Profile Reductions Dots)
noncomputable section

/-- Actual original-grid source execution, interpreted with the selected SM80
instruction family. All returned bytes, the full exterior byte frame, and every
original input backing byte are derived. Numerical ports are scalar operations,
scalar conversions and a fixed K8 MMA lane; no compiler/output premise occurs. -/
theorem rms_rbe_matmul_repaired_fp_returns
    (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (i : Inputs) (h : InputContract i) (a : Checked.Allocation i) (returns : Checked.Returns i a)
    (cast : Cast) (before prepared : Bytes) (prep : KernelBinding.Preparation i a cast before prepared)
    (scratch : State Term) :
    Contract.LaunchAdmitted i ∧
    (Contract.checkInputs i=.ok () ↔ InputContract i) ∧
    returns.tensor.metadata.shape=
      [(Checked.shape i).B,(Checked.shape i).M,(Checked.shape i).heads,(Checked.shape i).headDim] ∧
    returns.tensor.metadata.dtype=i.weight.metadata.dtype ∧
    returns.tensor.metadata.device=i.x.metadata.device ∧
    returns.tensor.metadata.dataPtr=a.out.metadata.dataPtr ∧
    ∃ final, KernelGrid.launch (KernelNative.operations R D) (fp8 i) (Checked.shape i).B (parameters i)
      (KernelBinding.state i a prepared scratch)=some final ∧
      (∀ x : KernelNativeComplete.Coordinate i, ∀ k : Fin (KernelPhysical.storeWidth i),
        KernelPhysical.outputBytes I P.schedule i a prepared final
          ((RmsMatmulRbe.Checked.pointer64 returns.tensor (KernelGrid.flat (parameters i) x)).toNat+k.val)=
          KernelPhysical.byteOf (KernelNativeComplete.reference I P R D i a cast before x) k.val) ∧
      (∀ byte, KernelPhysical.Outside i a byte →
        KernelPhysical.outputBytes I P.schedule i a prepared final byte=prepared byte) ∧
      (∀ t, t∈i.reads → ∀ byte, byte<t.bytes →
        KernelPhysical.outputBytes I P.schedule i a prepared final (t.base+byte)=before (t.base+byte)) := by
  obtain ⟨hs,hd,hdev,hptr,_⟩ := KernelTensorMetadata.returned i a returns
  exact ⟨h.2,Contract.checkInputs_iff i,hs,hd,hdev,hptr,
    KernelNativeComplete.complete I P R D i h.1 a returns cast before prepared prep scratch⟩

theorem rms_rbe_matmul_native_program_ids (i : Inputs) (h : InputContract i) (q : KernelGrid.Program)
    (member : q ∈ KernelGrid.programs (Checked.shape i).B (KernelBinding.parameters i)) :
    (BitVec.ofNat 32 q.1).toInt = (q.1 : Int) ∧
    (BitVec.ofNat 32 q.2).toInt = (q.2 : Int) :=
  Launch.native_program_ids i h q member

#axiomsClean rms_rbe_matmul_native_program_ids
#axiomsClean rms_rbe_matmul_repaired_fp_returns
#axiomsClean KernelPhysical.store_bytes
#specNonCircular KernelNativeComplete.reference avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGrid.result,KernelGrid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.TensorBridge

namespace VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.TensorBridge
open VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.Contract (Inputs InputContract)
noncomputable section

theorem python_checks_iff_input_contract (i : Inputs) :
    Contract.checkInputs i=.ok () ↔ InputContract i := Contract.checkInputs_iff i

#axiomsClean python_checks_iff_input_contract
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmulLaunchGuard.TensorBridge
