import bench.repairs.tritonbench_g.rowwise_quantization_triton.Kernels
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Contract
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Native
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Encoding
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.RealGrid

/-! Public independent real mathematics and complete checked output-pair
contract. Exact native values and numerical equivalence remain in FPEquiv. -/
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.HistoricalSpecification
noncomputable section
open Direct

def mathematical (values : Nat → Nat → ℝ) (rows columns : Nat)
    (quantized : Nat → Nat → Byte) (maximum : Nat → ℝ) : Prop :=
  (∀ r,r<rows → maximum r=logicalMaximum (values r) columns) ∧
  (∀ r,r<rows → ∀ c,c<columns → quantized r c=mathematicalByte (values r) columns c)

/-- Independent row maximum and floor/parity nearest-even rule, including zero
rows. The second stored output is the separately stated FP16 cast of maximum. -/
theorem real_rows (half : ℝ → Half) (p : Launch) (values : Nat → Nat → ℝ) :
    mathematical values p.rows p.columns
      (fun r c => rowQuantized (idealOperations half) (values r c)
        (rowMaximum (idealOperations half) p (values r)))
      (fun r => rowMaximum (idealOperations half) p (values r)) :=
  ⟨fun r _ => source_maximum_is_logical half p (values r),
    fun r _ c hc => source_quantization_is_logical half p (values r) c hc⟩

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInputs x=.ok () ↔ AdmittedArguments x := checkInputs_iff x

/-- Both returned Tensors, every int8 output and FP16 maximum byte, and the
frame of every preexisting live byte. Inputs are directly pointer-bound. -/
theorem rowwise_wrapper_correct (allocate : Allocator) (ops : Operations Word)
    (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (before : State) (resident : Resident x laws h before) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover))
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (success : checkedForward allocate ops x laws h before resident tree cover schedule=.ok (some (q,m,after))) :
    ResultCorrect ops (Direct.launch x.metadata h tree cover) (input x laws before) before q m after :=
  checked_correct allocate ops x laws h before resident tree cover schedule q m after success


/-- Complete checked ideal-real wrapper, both returned allocations, all int8
and FP16 output bytes, and caller frame. Real inputs are arbitrary storage
values and never encoded into Word. The scalar half cast remains explicit. -/
theorem rowwise_real_wrapper_correct (allocate : Allocator) (half : ℝ → Half)
    (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (storage : Nat → ℝ) (before : State) (tree : Tree)
    (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover))
    (device : Direct.OnDevice allocate x.metadata.device)
    (q m : RowwiseQuantizationTriton.Tensor) (after : Heap)
    (success : RealGrid.checkedForward allocate half x h storage before tree cover schedule=.ok (some (q,m,after))) :
    RealGrid.Result half (Direct.launch x.metadata h tree cover) (RealGrid.inputValues storage x) before q m after ∧
    q.device=x.metadata.device ∧ m.device=x.metadata.device ∧
    (∀ r,r<rows x.metadata → ∀ c,c<columns x.metadata →
      storage ((Direct.view x laws h).base+4*(r*columns x.metadata+c))=RealGrid.inputValues storage x r c) := by
  refine ⟨RealGrid.checked_correct allocate half x h storage before tree cover schedule q m after success,?_,?_,?_⟩
  · exact (RealGrid.checked_devices allocate half x h storage before tree cover schedule device q m after success).1
  · exact (RealGrid.checked_devices allocate half x h storage before tree cover schedule device q m after success).2
  · intro r hr c hc
    exact RealGrid.input_pointer storage x laws h r c hr hc

#axiomsClean rowwise_real_wrapper_correct

#specNonCircular mathematical avoiding [sourceRow,runRow,runGrid,Direct.forward,checkedForward]
#axiomsClean real_rows
#axiomsClean rowwise_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.HistoricalSpecification
