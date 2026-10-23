import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelRealComplete
import bench.repairs.tritonbench_g.rms_rbe_matmul.Proofs.KernelTensorMetadata

namespace VeriTile.Bench.Repairs.RmsRbeMatmul
open VeriTile.Triton FP.Structural
open Contract (Inputs InputContract)
open KernelBinding (Bytes Cast Preparation parameters fp8)
open KernelNative (Word)
noncomputable section

theorem python_checks_iff_input_contract (i : Inputs) :
    Contract.checkInputs i=.ok () ↔ InputContract i := Contract.checkInputs_iff i

/-- Whole returned Tensor under exact-Real arithmetic, derived from actual
source execution on the original rectangular grid. The mathematical target
reads only original input words and their scalar RMS conversion. -/
specification rms_rbe_matmul_repaired_real_returns
    (d32 d16 : Word → ℝ) (d8 : Float8E5Bits → ℝ) (zero : d8 (Float8E5Bits.ofInt 0)=0)
    (i : Inputs) (h : InputContract i) (a : Checked.Allocation i) (returns : Checked.Returns i a)
    (cast : Cast) (before prepared : Bytes) (prep : KernelBinding.Preparation i a cast before prepared)
    (scratch : State ℝ) :
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
  exact ⟨hs,hd,hdev,hptr,KernelRealComplete.complete d32 d16 d8 zero i h a cast before prepared prep scratch⟩

#axiomsClean python_checks_iff_input_contract
#axiomsClean rms_rbe_matmul_repaired_real_returns
#specNonCircular KernelRealComplete.target avoiding [FP.Structural.exec,KernelSource.inputs,
  KernelSource.result,KernelSource.scatter,KernelGraph.output,KernelGrid.result,KernelGrid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsRbeMatmul
