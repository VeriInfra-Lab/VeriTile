import bench.repairs.tritonbench_g.f8_conversion_utils.Misc.NativeCorrect
import bench.repairs.tritonbench_g.f8_conversion_utils.Correct
import bench.repairs.tritonbench_g.f8_conversion_utils.FPEquiv
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.LogicalStorage

/-! Optional actual Tensor applications, wrapper outcomes and checked pointer
bindings. Identical helper statements formerly exposed by both primary entries
are shared here as ordinary theorems. No input or numerical domain changes. -/
namespace VeriTile.Bench.Repairs.F8ConversionUtils.TensorBridge
open VeriTile.Torch Checked
noncomputable section
variable {d : Direction} {f : FormatArgument} {x : TensorMetadata} {identity : Nat}

theorem checked_wrapper_real_correctness (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (before : Cells.Heap) (savedDevice : Nat)
    (ready : Bool) (launch : x.numel = 0 ∨ ready = Bool.true) :
    InputContract d f x ∧
    ∃ result,runChecked (Cells.primitives _ _ a.source.positive a.target.positive)
        id before savedDevice (.ok a) ready = .ok result ∧
      result.tensor = a.result ∧ MetadataSpec identity d x result.tensor ∧
      result.currentDevice = savedDevice ∧
      (∀ i,i<x.numel → result.memory (Memory.address a.target.base (profile d f x).outputWidth i) =
        before (Memory.address x.dataPtr.toNat (profile d f x).inputWidth i)) ∧
      (∀ q,¬a.target.Active q → result.memory q=before q) ∧
      RealCalls (profile d f x) a.source a.target :=
  Checked.checked_real checked a before savedDevice ready launch

theorem native_input_pointer (a : Allocated identity d f x) (c pid lane : Nat)
    (launched : pid<NativeIndex.grid (Chunking.chunkLength x.numel c)) (inTile : lane<1024)
    (active : pid*1024+lane<Chunking.chunkLength x.numel c) :
    NativeIndex.pointer (a.source.base+Chunking.chunkStart c*(profile d f x).inputWidth)
      (profile d f x).inputWidth pid lane =
      Memory.address x.dataPtr.toNat (profile d f x).inputWidth
        (Chunking.chunkStart c+(pid*1024+lane)) :=
  Checked.input_pointer a c pid lane launched inTile active

theorem wrapper_outcomes {M A B : Type} (ops : Memory.Primitives M A B
    (profile d f x).inputWidth (profile d f x).outputWidth)
    (compute : A → B) (before : M) (savedDevice : Nat)
    (checked : checkInputs d f (.tensor x) = .ok ())
    (allocated : Except Unit (Allocated identity d f x)) (ready : Bool) :
    (allocated = .error () ∧ runChecked ops compute before savedDevice allocated ready = .error .allocation) ∨
    (∃ a,allocated = .ok a ∧ x.numel ≠ 0 ∧ ready = Bool.false ∧
      runChecked ops compute before savedDevice allocated ready = .error .launch) ∨
    (∃ a,allocated = .ok a ∧ (x.numel = 0 ∨ ready = Bool.true) ∧
      runChecked ops compute before savedDevice allocated ready =
        .ok ⟨a.result,Lowering.allChunks ops (profile d f x).twice compute
          a.source.base a.target.base x.numel (launchChunks x.numel) before,savedDevice⟩) :=
  Checked.outcomes ops compute before savedDevice checked allocated ready

theorem checked_input_guard (d : Direction) (f : FormatArgument) (x : ArgumentMetadata) :
    checkInputs d f x = .ok () ↔ AdmittedArguments d f x := Checked.checkInputs_iff d f x

theorem native_alignment (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (c pid lane : Nat)
    (launched : pid<NativeIndex.grid (Chunking.chunkLength x.numel c)) (inTile : lane<1024)
    (active : pid*1024+lane<Chunking.chunkLength x.numel c) :
    NativeIndex.pointer (a.source.base+Chunking.chunkStart c*(profile d f x).inputWidth)
      (profile d f x).inputWidth pid lane % (profile d f x).inputWidth = 0 ∧
    NativeIndex.pointer (a.target.base+Chunking.chunkStart c*(profile d f x).outputWidth)
      (profile d f x).outputWidth pid lane % (profile d f x).outputWidth = 0 :=
  Checked.accessed_alignment checked a c pid lane launched inTile active


/-- The same source and physical storage binding also implement the independent
Real cast-erasure formula at every logical Tensor coordinate. -/
theorem logical_tensor_real_correctness
    (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x)
    (input : TensorStorage (profile d f x).inputWidth x)
    (output : TensorStorage (profile d f x).outputWidth a.result.metadata)
    (before : Cells.Heap) :
    ∀ i : Fin x.numel,
      logicalReal output a.noNeg
        (Lowering.allChunks (Cells.primitives _ _ a.source.positive a.target.positive)
          (profile d f x).twice id a.source.base a.target.base x.numel (launchChunks x.numel) before)
        (resultIndex a i) = conversionReal (logicalReal input ((checkInputs_iff d f (.tensor x)).mp checked).noNeg before i) :=
  logical_real_result checked a input output before

theorem checked_wrapper_fp_equivalence (checked : checkInputs d f (.tensor x) = .ok ())
    (a : Allocated identity d f x) (before : Bytes.Heap) (savedDevice : Nat)
    (policy : Numerics.Policy)
    (ready : Bool) (launch : x.numel = 0 ∨ ready = Bool.true) :
    InputContract d f x ∧
    ∃ result,runChecked (Bytes.primitives _ _) (conversionBits policy (profile d f x)) before savedDevice (.ok a) ready = .ok result ∧
      result.tensor = a.result ∧ MetadataSpec identity d x result.tensor ∧
      result.currentDevice = savedDevice ∧
      (∀ i,i<x.numel → Bytes.read (profile d f x).outputWidth result.memory
          (Memory.address a.target.base (profile d f x).outputWidth i) =
        Numerics.reference policy (profile d f x)
          (Bytes.read (profile d f x).inputWidth before
            (Memory.address x.dataPtr.toNat (profile d f x).inputWidth i))) ∧
      (∀ q,¬a.target.Active q → result.memory q=before q) ∧
      FPCalls (profile d f x) (conversionBits policy (profile d f x)) a.source a.target := by
  obtain ⟨contract,result,executed,tensor,metadata,device,values,frame,_,calls⟩ :=
    Checked.checked_fp checked a before savedDevice (conversionBits policy (profile d f x))
      policy (fun _ => rfl) ready launch
  exact ⟨contract,result,executed,tensor,metadata,device,values,frame,calls⟩

/-- Source execution is related to the independent logical Tensor formula,
including every output byte, exterior memory, and every actual chunk program. -/
theorem logical_tensor_bit_correctness
    (checked : checkInputs d f (.tensor x) = .ok ())
    (policy : Numerics.Policy) (a : Allocated identity d f x)
    (input : TensorStorage (profile d f x).inputWidth x)
    (output : TensorStorage (profile d f x).outputWidth a.result.metadata)
    (before : Bytes.Heap) :
    (∀ i : Fin x.numel,
      logicalBits output a.noNeg (bitResult policy a before) (resultIndex a i) =
        conversionBits policy (profile d f x) (logicalBits input ((checkInputs_iff d f (.tensor x)).mp checked).noNeg before i)) ∧
    (∀ q,¬a.target.Active q → bitResult policy a before q=before q) ∧
    FPCalls (profile d f x) (conversionBits policy (profile d f x)) a.source a.target :=
  logical_bit_result checked policy a input output before

#axiomsClean checked_wrapper_real_correctness
#axiomsClean native_input_pointer
#axiomsClean wrapper_outcomes
#axiomsClean checked_input_guard
#axiomsClean native_alignment
#axiomsClean logical_tensor_real_correctness
#axiomsClean checked_wrapper_fp_equivalence
#axiomsClean logical_tensor_bit_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.F8ConversionUtils.TensorBridge
