import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Correct
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.StoreABI
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.BoundBridge
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TorchInput

/-! Optional shared Torch Tensor application and signed-representability
corollary. The complete common-Tensor/backing applications are retained here. The raw
primary grid results use the same grid_low32/grid_values/grid_frame certificates
and nativeAccumulation_low32 recurrence, with no independently reproved source execution. -/
namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel TensorModel
noncomputable section
set_option maxHeartbeats 8000000

/-- The exact readable quotation is executed through its explicit finite
compile-assertion projection. All source programs run on evolving common
memory. Every Tensor cell reads the independent low32 dense signed sum from
that same backing memory, and all observations outside C's cells are framed. -/
theorem kernel_tensor_low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (rest : Algebra Word32) (request : Request) (aPointer bPointer out : Nat)
    (before : Memory Word32) (admitted : InputContract request aPointer bPointer before)
    (separate : out ≠ request.a.storage ∧ out ≠ request.b.storage)
    (c : Fin 15) (legal : request.shape.legal c) :
    GridExec cast rest (parameters (layout request) request.shape c)
      (encodedState request out before) (List.range (request.shape.programs c))
      (flatten (encodedState request out before))
      (flatten (encodedState request out (resultMemory request out before c))) ∧
    (∀ m n, m < request.shape.m → n < request.shape.n →
      tensorRead32 (outputTensor request out) (resultMemory request out before c) m n =
        tensorWordTarget request out before m n) ∧
    (∀ loc, (loc.1 ≠ (out,.int32) ∨ request.shape.m*request.shape.n ≤ loc.2) →
      resultMemory request out before c loc = before loc) ∧
    SourceBounds request.shape (hostBuffers request out) := by
  have domains := source_domains request out before (admitted_defined request aPointer bPointer out before admitted)
  refine ⟨?_,tensor_readback request out before c,backing_frame request out before c,
    admitted_indices request aPointer bPointer out before admitted⟩
  rw [encode_result request out before separate c]
  exact grid_low32 cast castLeaf rest (layout request) request.shape c legal
    (encodedState request out before) domains.1 domains.2 (by exact ⟨rfl,rfl⟩)
    (by change outputPort ≠ inputPort ∧ outputPort ≠ weightPort; decide)

/-- The numeric native binding is confined to fixed FP32→signed8, K32 MMA,
and two-word add.u32 leaves. Actual casts, packing, block order and stores are
connected by the quoted-source proofs, not assumed by a kernel-level axiom. -/
theorem native_tensor_lane (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (leaves : NumericLeaves cast instruction)
    (addLeaf : Add32Leaf add32) (request : Request) (aPointer bPointer out : Nat)
    (before : Memory Word32) (admitted : InputContract request aPointer bPointer before)
    (c : Fin 15) (legal : request.shape.legal c) (m n : Nat)
    (hm : m < request.shape.m) (hn : n < request.shape.n) :
    tensorRead32 (outputTensor request out) (resultMemory request out before c) m n =
      nativeModularAccumulation cast instruction add32 request.shape c
        (fun k => before (aAddress (hostBuffers request out) m k))
        (fun q => before (bAddress (hostBuffers request out) q n)) := by
  rw [tensor_readback request out before c m n hm hn]
  symm
  exact nativeAccumulation_low32 cast instruction add32 leaves addLeaf request.shape c legal _ _
    (admitted_defined request aPointer bPointer out before admitted m hm)

/-- Full quoted-source grid execution and primitive-native numeric agreement
share the same initial/final backing encoding and Tensor readback. Compiler
correspondence remains scoped to the two separately pinned SM90 profiles. -/
theorem native_tensor_kernel (cast : Word32 → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word32 → Word32)
    (add32 : Word32 → Word32 → Word32) (leaves : NumericLeaves cast instruction)
    (addLeaf : Add32Leaf add32) (rest : Algebra Word32)
    (request : Request) (aPointer bPointer out : Nat) (before : Memory Word32)
    (admitted : InputContract request aPointer bPointer before)
    (separate : out ≠ request.a.storage ∧ out ≠ request.b.storage)
    (c : Fin 15) (legal : request.shape.legal c) :
    GridExec cast rest (parameters (layout request) request.shape c)
      (encodedState request out before) (List.range (request.shape.programs c))
      (flatten (encodedState request out before))
      (flatten (encodedState request out (resultMemory request out before c))) ∧
    (∀ m n, m < request.shape.m → n < request.shape.n →
      tensorRead32 (outputTensor request out) (resultMemory request out before c) m n =
        nativeModularAccumulation cast instruction add32 request.shape c
          (fun k => before (aAddress (hostBuffers request out) m k))
          (fun q => before (bAddress (hostBuffers request out) q n))) ∧
    (∀ loc, (loc.1 ≠ (out,.int32) ∨ request.shape.m*request.shape.n ≤ loc.2) →
      resultMemory request out before c loc = before loc) := by
  obtain ⟨run,_,frame,_⟩ := kernel_tensor_low32 cast leaves.cast rest request aPointer bPointer out
    before admitted separate c legal
  exact ⟨run,native_tensor_lane cast instruction add32 leaves addLeaf request aPointer bPointer out
    before admitted c legal,frame⟩

/-- The native word target also reads A through the actual shared FP32 Tensor
logical API and the same common backing; B/C keep their ordinary UINT8/INT32
descriptors. This corollary is downstream of the source execution theorem. -/
theorem kernel_torch_low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (rest : Algebra Word32) (request : Request) (aPointer bPointer out : Nat)
    (before : Memory Word32) (admitted : InputContract request aPointer bPointer before)
    (binding : FP32Binding request.a aPointer)
    (separate : out ≠ request.a.storage ∧ out ≠ request.b.storage)
    (c : Fin 15) (legal : request.shape.legal c) :
    GridExec cast rest (parameters (layout request) request.shape c)
      (encodedState request out before) (List.range (request.shape.programs c))
      (flatten (encodedState request out before))
      (flatten (encodedState request out (resultMemory request out before c))) ∧
    (∀ m n, m < request.shape.m → n < request.shape.n →
      tensorRead32 (outputTensor request out) (resultMemory request out before c) m n =
        BitVec.ofInt 32 (torchSignedSum request aPointer out before binding m n)) ∧
    (∀ loc, (loc.1 ≠ (out,.int32) ∨ request.shape.m*request.shape.n ≤ loc.2) →
      resultMemory request out before c loc = before loc) := by
  obtain ⟨run,values,frame,_⟩ := kernel_tensor_low32 cast castLeaf rest request aPointer bPointer out
    before admitted separate c legal
  refine ⟨run,?_,frame⟩
  intro m n hm hn
  rw [torch_target request aPointer bPointer out before binding admitted m n hm]
  exact values m n hm hn

/-- Final signed representability is a conclusion-changing corollary premise,
never an input guard or a substitute for modular native correctness. -/
theorem kernel_tensor_representable (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15)
    (m n : Nat) (hm : m < request.shape.m) (hn : n < request.shape.n)
    (lo : -2147483648 ≤ tensorSignedSum request out before m n)
    (hi : tensorSignedSum request out before m n ≤ 2147483647) :
    (tensorRead32 (outputTensor request out) (resultMemory request out before c) m n).toInt =
      tensorSignedSum request out before m n := by
  rw [tensor_readback request out before c m n hm hn]
  unfold tensorWordTarget
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

#axiomsClean kernel_tensor_low32
#axiomsClean native_tensor_lane
#axiomsClean native_tensor_kernel
#axiomsClean kernel_torch_low32
#axiomsClean kernel_tensor_representable
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorBridge
