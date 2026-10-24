import bench.repairs.tritonbench_g.rowwise_quantization_triton.Correct
import bench.repairs.tritonbench_g.rowwise_quantization_triton.FPEquiv
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.KernelTensor
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Native

/-! Optional input-check correspondence and Tensor binding helpers. The
complete accepted kernel specifications remain in the primary entries. -/
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton.TensorBridge
open VeriTile.Triton KernelNativeProfile KernelTensor
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInputs x = .ok () ↔ AdmittedArguments x := checkInputs_iff x

/-- The FP32 input's observed pointer, checked alignment and full byte span
are connected to the same logical Tensor reader used in the result theorem. -/
theorem input_pointer_refinement (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (r c : Nat)
    (hr : r < rows x.metadata) (hc : c < columns x.metadata) :
    (x.dataPtr + Native.nativeOffset r (columns x.metadata) c * BitVec.ofNat 64 4).toNat =
      (Direct.view x laws h).base + 4*(r*columns x.metadata+c) ∧
    (Direct.view x laws h).base % 4 = 0 ∧
    (Direct.view x laws h).base + 4*(r*columns x.metadata+c) + 3 < 2^64 :=
  KernelTensor.input_pointers x laws h r c hr hc

/-- Ordinary output allocation facts connect the actual observed pointers to
the signed64 source arithmetic and each Tensor's physical logical index. -/
theorem output_pointer_refinement (x : Direct.Tensor) (laws : Direct.TensorLaws x)
    (h : InputContract x.metadata) (q m : OutputTensor)
    (ql : OutputLaws q 1 (.other "int8") [rows x.metadata, columns x.metadata] x.metadata.device)
    (ml : OutputLaws m 2 (.other "float16") [rows x.metadata] x.metadata.device)
    (r c : Nat) (hr : r < rows x.metadata) (hc : c < columns x.metadata) :
    (q.metadata.dataPtr + Native.nativeOffset r (columns x.metadata) c * BitVec.ofNat 64 1).toNat =
      address q 1 (r*columns x.metadata+c) ∧
    (m.metadata.dataPtr + (BitVec.ofNat 32 r).signExtend 64 * BitVec.ofNat 64 2).toNat =
      address m 2 r ∧
    address q 1 (r*columns x.metadata+c) < 2^64 ∧ address m 2 r + 1 < 2^64 ∧
    address m 2 r % 2 = 0 :=
  KernelTensor.output_pointers x laws h q m ql ml r c hr hc

/-- Independent maximum and nearest-even int8 formula for every logical row.
The quoted complete kernel executes with the documented task-local llrint and
named-register extension. Real interprets floating arithmetic and storage
casts mathematically; native FP32/FP16 rounding is stated in FPEquiv. -/
theorem rowwise_kernel_correct (x : TensorMetadata) (h : InputContract x)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x)))
    (schedule : Schedule (Direct.launch x h tree cover))
    (values : Nat → Nat → ℝ) (rest : FP.Structural.Algebra ℝ)
    (initial : FP.Structural.State ℝ)
    (reads : KernelWhole.Reads (Direct.launch x h tree cover) values initial) :
    Spec.Real (∃ final,
      KernelGrid.run (KernelRealResult.algebra (Direct.launch x h tree cover) rest)
        KernelRealResult.round (KernelProfile.equal KernelRealResult.ops)
        (Direct.launch x h tree cover) schedule.order initial = some final ∧
      KernelRealResult.Result (Direct.launch x h tree cover) values initial final) :=
  KernelRealResult.kernel_correct rest (Direct.launch x h tree cover) schedule values initial reads

/-- Exact source-order FP32 result, signed64 llrint, INT8 codes, and FP16 maxima.
All FP32 inputs remain admitted. The result executes the complete quoted
source through the explicitly documented local interpreter extension, then
connects its stores to both actual Tensor storage addresses and byte frames.
There is no assumed successful run, output equation, or numerical agreement. -/
theorem rowwise_fp32 (a : Arithmetic) (rest : FP.Structural.Algebra Word)
    (x : Direct.Tensor) (laws : Direct.TensorLaws x) (h : InputContract x.metadata)
    (tree : Tree) (cover : tree.leaves.Perm (List.range (paddedWidth x.metadata)))
    (schedule : Schedule (Direct.launch x.metadata h tree cover)) (q m : OutputTensor)
    (ql : OutputLaws q 1 (.other "int8") [rows x.metadata, columns x.metadata] x.metadata.device)
    (ml : OutputLaws m 2 (.other "float16") [rows x.metadata] x.metadata.device)
    (sep : Separation (Direct.launch x.metadata h tree cover) (layout x laws h q m))
    (heap : Heap) (scratch : FP.Structural.State Word) :
    ∃ final,
      KernelGrid.run (algebra a (Direct.launch x.metadata h tree cover) rest) (llrint a) (equal a)
        (Direct.launch x.metadata h tree cover) schedule.order (initial x laws h heap scratch) = some final ∧
      KernelNativeResult.Result a (Direct.launch x.metadata h tree cover) (values x laws heap)
        (initial x laws h heap scratch) final ∧
      ByteResult a x laws h tree cover q m heap (byteExecution a x laws h tree cover schedule q m heap) ∧
      (∀ r c, r < rows x.metadata → c < columns x.metadata →
        byteExecution a x laws h tree cover schedule q m heap (address q 1 (r*columns x.metadata+c)) =
          narrow8 ((final.mem "q" (r*columns x.metadata+c)).read .int)) ∧
      (∀ r, r < rows x.metadata → ∀ b : Fin 2,
        byteExecution a x laws h tree cover schedule q m heap (address m 2 r + b.val) =
          halfByte (KernelEncoding.unpackHalf ((final.mem "m" r).read .fp16)) b) :=
  KernelTensor.actual_kernel a rest x laws h tree cover schedule q m ql ml sep heap scratch

/-- The same checked domain makes every source integer offset exact, including
padded inactive lanes. This is finite machine arithmetic, not Real indexing
assumed to match the source. -/
theorem source_index_refinement (x : TensorMetadata) (h : InputContract x)
    (r lane : Nat) (hr : r < rows x) (hl : lane < paddedWidth x) :
    (Native.nativeOffset r (columns x) lane).toInt = (r*columns x+lane : Nat) :=
  Native.offset_exact x h r lane hr hl

#axiomsClean rowwise_kernel_correct
#axiomsClean rowwise_fp32
#axiomsClean source_index_refinement
#axiomsClean python_checks_iff_preconditions
#axiomsClean input_pointer_refinement
#axiomsClean output_pointer_refinement
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton.TensorBridge
