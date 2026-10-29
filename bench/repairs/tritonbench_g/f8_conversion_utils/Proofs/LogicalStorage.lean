import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Checked
open VeriTile.Torch
noncomputable section
variable {d : Direction} {f : FormatArgument} {x : TensorMetadata} {identity : Nat}

theorem logical_address (t : TensorStorage width metadata)
    (contiguous : metadata.isContiguous = Bool.true) (i : Fin metadata.numel) :
    t.address i = Memory.address metadata.dataPtr.toNat width i.val := by
  rw [TensorStorage.address,t.contiguous contiguous i,t.pointer]
  simp [Memory.address,Nat.add_mul,Nat.add_assoc]

def resultIndex (a : Allocated identity d f x) (i : Fin x.numel) : Fin a.result.metadata.numel :=
  ⟨i.val,by simpa only [TensorMetadata.numel,a.sameShape] using i.isLt⟩

def bitResult (policy : Numerics.Policy) (a : Allocated identity d f x) (before : Bytes.Heap) : Bytes.Heap :=
  Lowering.allChunks (Bytes.primitives _ _) (profile d f x).twice
    (conversionBits policy (profile d f x)) a.source.base a.target.base x.numel
    (launchChunks x.numel) before

/-- Logical coordinates are connected to actual initial bytes and actual
final native memory. The guard proves contiguity/no-negation; storage facts
only state the ordinary Tensor API meaning of data_ptr and logical indices. -/
theorem logical_bit_result (checked : checkInputs d f (.tensor x) = .ok ())
    (policy : Numerics.Policy) (a : Allocated identity d f x)
    (input : TensorStorage (profile d f x).inputWidth x)
    (output : TensorStorage (profile d f x).outputWidth a.result.metadata)
    (before : Bytes.Heap) :
    (∀ i : Fin x.numel,
      logicalBits output a.noNeg (bitResult policy a before) (resultIndex a i) =
        conversionBits policy (profile d f x) (logicalBits input ((checkInputs_iff d f (.tensor x)).mp checked).noNeg before i)) ∧
    (∀ q,¬a.target.Active q → bitResult policy a before q=before q) ∧
    FPCalls (profile d f x) (conversionBits policy (profile d f x)) a.source a.target := by
  have contract := (checkInputs_iff d f (.tensor x)).mp checked
  constructor
  · intro i
    unfold logicalBits
    rw [logical_address output a.contiguous,logical_address input contract.contiguous]
    simp only [resultIndex,←a.outputPointer,bitResult,Lowering.complete]
    have value := (Native.full_execution (Bytes.primitives _ _) (profile d f x).twice
      (conversionBits policy (profile d f x)) a.source a.target a.separate before).1 i.val i.isLt
    simpa only [Bytes.primitives,a.inputPointer] using value
  · constructor
    · intro q outside
      unfold bitResult
      rw [Lowering.complete]
      exact Bytes.byte_eq _ _ _ _ q
        ((Native.full_execution (Bytes.primitives _ _) (profile d f x).twice
          (conversionBits policy (profile d f x)) a.source a.target a.separate before).2 q outside)
    · exact KernelContract.native_calls policy _ a.source a.target

theorem logical_real_result (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x)
    (input : TensorStorage (profile d f x).inputWidth x)
    (output : TensorStorage (profile d f x).outputWidth a.result.metadata)
    (before : Cells.Heap) :
    ∀ i : Fin x.numel,
      logicalReal output a.noNeg
        (Lowering.allChunks (Cells.primitives _ _ a.source.positive a.target.positive)
          (profile d f x).twice id a.source.base a.target.base x.numel (launchChunks x.numel) before)
        (resultIndex a i) = conversionReal (logicalReal input ((checkInputs_iff d f (.tensor x)).mp checked).noNeg before i) := by
  intro i
  have contract := (checkInputs_iff d f (.tensor x)).mp checked
  unfold logicalReal conversionReal
  rw [logical_address output a.contiguous,logical_address input contract.contiguous]
  simp only [resultIndex,←a.outputPointer,Lowering.complete]
  simpa only [a.inputPointer] using (Cells.complete (profile d f x).twice
    a.source a.target a.separate before).1 i.val i.isLt

#axiomsClean logical_address
#axiomsClean logical_bit_result
#axiomsClean logical_real_result
end
end VeriTile.Bench.Repairs.F8ConversionUtils.Checked
