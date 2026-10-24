import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelNativeComplete
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelTensorMetadata

namespace VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open Contract (Inputs InputContract)
open KernelBinding (Bytes Cast Preparation parameters fp8)
open KernelNative (Term Instructions Profile Reductions Dots)
noncomputable section

/-- Actual original-grid source execution, interpreted with the selected SM80
instruction family. All returned bytes, the full exterior byte frame, and every
original input backing byte are derived. Numerical ports are scalar operations,
scalar conversions and a fixed K8 MMA lane; no compiler/output premise occurs. -/
specification rms_rbe_matmul_repaired_fp_returns
    (I : Instructions) (P : Profile) (R : Reductions) (D : Dots)
    (i : Inputs) (h : InputContract i) (a : Checked.Allocation i) (returns : Checked.Returns i a)
    (cast : Cast) (before prepared : Bytes) (prep : KernelBinding.Preparation i a cast before prepared)
    (scratch : State Term) :
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
  exact ⟨Contract.checkInputs_iff i,hs,hd,hdev,hptr,
    KernelNativeComplete.complete I P R D i h a returns cast before prepared prep scratch⟩

open RmsMatmulRbe in
specification rms_rbe_matmul_native_addresses (i : RmsRbeMatmul.Contract.Inputs)
    (h : RmsRbeMatmul.Contract.InputContract i) (a : RmsRbeMatmul.Checked.Allocation i)
    (active : 0<(RmsRbeMatmul.Checked.shape i).B ∧ 0<(RmsRbeMatmul.Checked.shape i).M) :
    RmsMatmulRbe.Checked.ActiveBounds (RmsRbeMatmul.Checked.shape i) ∧ RmsRbeMatmul.Checked.LoadsValid i a ∧
    (∀ x : VeriTile.Bench.Repairs.RmsRbeMatmul.Cell (RmsRbeMatmul.Checked.shape i),RmsMatmulRbe.Checked.PointerValid a.out
      (RmsMatmulRbe.Checked.flat3 (RmsRbeMatmul.Checked.shape i).M (RmsRbeMatmul.Checked.shape i).N x.1 x.2.1 x.2.2)) ∧
    (i.rotary=Bool.true → ∀ row,row<tilesM (RmsRbeMatmul.Checked.shape i)*16 →
      -(2^63:Int) ≤ i.scalars.start+(row:Int) ∧ i.scalars.start+(row:Int)<(2^63:Int)) ∧
    (∀ x : VeriTile.Bench.Repairs.RmsRbeMatmul.Cell (RmsRbeMatmul.Checked.shape i),
      (RmsMatmulRbe.Checked.pointer64 a.out (RmsMatmulRbe.Checked.flat3
        (RmsRbeMatmul.Checked.shape i).M (RmsRbeMatmul.Checked.shape i).N x.1 x.2.1 x.2.2)).toNat=
        a.out.base+RmsMatmulRbe.Contract.width a.out.metadata.dtype*(outputAddress (RmsRbeMatmul.Checked.buffers i a) x).2) :=
  ⟨RmsRbeMatmul.Checked.active_bounds i h a active,
    RmsRbeMatmul.Checked.loads_valid i h a,
    RmsRbeMatmul.Checked.output_pointer_valid i a,
    fun rotary => RmsRbeMatmul.Checked.padded_position i h rotary active,RmsRbeMatmul.Checked.output_address64 i a⟩

#axiomsClean rms_rbe_matmul_native_addresses
#axiomsClean rms_rbe_matmul_repaired_fp_returns
#axiomsClean KernelPhysical.store_bytes
#specNonCircular KernelNativeComplete.reference avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGrid.result,KernelGrid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmul
