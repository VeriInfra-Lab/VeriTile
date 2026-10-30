import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelEntries
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelBounds
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelResult
open VeriTile.Triton FP.Structural KernelTensor KernelEntries
noncomputable section
set_option maxHeartbeats 500000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem native_readback (p : Prepared) (before : KernelStorage.Heap) (final : State Word)
    (w : Work) (hw : p.config.Active w) (value : Word)
    (written : ∀ byte : Fin 4,nativeHeap p before final
      (p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val)=
        KernelStorage.bytes value byte) :
    nativeResult p before final w.1 w.2=value := by
  simp only [nativeResult,KernelTensor.nativePoint,VeriTile.Torch.FP32.logicalValue,
    p.c.notNeg,Bool.false_eq_true,ite_false,cCols p,p.valid.cShape.2,KernelStorage.read]
  have bytes : (fun byte : Fin 4=>nativeHeap p before final
      (p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val))=
        Physical.encode value.bits := by funext byte;exact written byte
  rw [bytes,Physical.decode_encode]

/-- Actual shared Tensor values, all four bytes, and every untouched physical
byte follow from execution of the full source and the store interpretation. -/
theorem native (instructions : KernelNative.Instructions) (p : Prepared) (before : KernelStorage.Heap) :
    ∃ final,KernelSource.launch KernelNative.scalar (KernelScalar.dotAcc (KernelNative.dot instructions p.config.opcode))
      p.config (initial p (KernelStorage.read before))=some final ∧
      (∀ w,p.config.Active w → nativeResult p before final w.1 w.2=nativeTarget instructions p before w.1 w.2) ∧
      (∀ w,p.config.Active w → ∀ byte : Fin 4,nativeHeap p before final
        (p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val)=
          KernelStorage.bytes (nativeTarget instructions p before w.1 w.2) byte) ∧
      (∀ q,(∀ w,p.config.Active w → ∀ byte : Fin 4,
        q≠p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val) →
        nativeHeap p before final q=before q) := by
  obtain ⟨t,he,hb,hf⟩:=KernelEntries.native_correct instructions p before
  refine ⟨t,he,?_,hb,hf⟩
  intro w hw
  exact native_readback p before t w hw _ (hb w hw)

theorem inputA_outside (p : Prepared) (r k : Nat) (hr : r<p.config.M) (hk : k<p.config.K)
    (byte : Fin 4) (w : Work) (hw : p.config.Active w) (outByte : Fin 4) :
    p.a.tensor.storage.address (p.a.tensor.physicalIndex (r*p.config.K+k))+byte.val≠
      p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+outByte.val := by
  have metadata:=((Checked.checkInputs_iff _).mp p.checked).1
  simp only [Checked.metadataChecks,decide_eq_true_eq] at metadata
  have positiveOut : 0<p.config.M*p.config.N :=
    Nat.mul_pos (Nat.lt_of_le_of_lt (Nat.zero_le _) hw.1) (Nat.lt_of_le_of_lt (Nat.zero_le _) hw.2)
  have positiveIn : 0<p.config.M*p.config.K :=
    Nat.mul_pos (Nat.lt_of_le_of_lt (Nat.zero_le _) hr) (Nat.lt_of_le_of_lt (Nat.zero_le _) hk)
  have distinct:=metadata.2.2.2.2.2.1 positiveOut positiveIn
  have disjoint:=p.separateA distinct
  have input:=active_bytes p.a r k (by simpa only [aRows p,p.valid.aShape.1] using hr)
    (by simpa only [aCols p,p.valid.aShape.2] using hk) byte
  have output:=active_bytes p.c w.1 w.2 (by simpa only [cRows p,p.valid.cShape.1] using hw.1)
    (by simpa only [cCols p,p.valid.cShape.2] using hw.2) outByte
  simp only [aCols p,p.valid.aShape.2,cCols p,p.valid.cShape.2] at input output
  rcases disjoint with disjoint|disjoint <;> omega

theorem inputB_outside (p : Prepared) (k n : Nat) (hk : k<p.config.K) (hn : n<p.config.N)
    (byte : Fin 4) (w : Work) (hw : p.config.Active w) (outByte : Fin 4) :
    p.b.tensor.storage.address (p.b.tensor.physicalIndex (k*p.config.N+n))+byte.val≠
      p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+outByte.val := by
  have metadata:=((Checked.checkInputs_iff _).mp p.checked).1
  simp only [Checked.metadataChecks,decide_eq_true_eq] at metadata
  have positiveOut : 0<p.config.M*p.config.N :=
    Nat.mul_pos (Nat.lt_of_le_of_lt (Nat.zero_le _) hw.1) (Nat.lt_of_le_of_lt (Nat.zero_le _) hw.2)
  have positiveIn : 0<p.config.K*p.config.N :=
    Nat.mul_pos (Nat.lt_of_le_of_lt (Nat.zero_le _) hk) (Nat.lt_of_le_of_lt (Nat.zero_le _) hn)
  have distinct:=metadata.2.2.2.2.2.2.1 positiveOut positiveIn
  have disjoint:=p.separateB distinct
  have input:=active_bytes p.b k n (by simpa only [bRows p,p.valid.bShape.1] using hk)
    (by simpa only [bCols p,p.valid.bShape.2] using hn) byte
  have output:=active_bytes p.c w.1 w.2 (by simpa only [cRows p,p.valid.cShape.1] using hw.1)
    (by simpa only [cCols p,p.valid.cShape.2] using hw.2) outByte
  simp only [bCols p,p.valid.bShape.2,cCols p,p.valid.cShape.2] at input output
  rcases disjoint with disjoint|disjoint <;> omega

#axiomsClean native_readback
#axiomsClean native
#axiomsClean inputA_outside
#axiomsClean inputB_outside
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelResult
