import bench.tritonbench_g.int8_matmul_kernel.Proofs.GridOutput
import bench.tritonbench_g.int8_matmul_kernel.Proofs.Mathematical_Bounds

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section
set_option maxHeartbeats 8000000

def inputPort : RegionName := ⟨"FP32_A"⟩
def weightPort : RegionName := ⟨"UINT8_B"⟩
def outputPort : RegionName := ⟨"INT32_C"⟩

def layout (request : Request) : Layout where
  input := inputPort
  weight := weightPort
  output := outputPort
  inputRowStride := request.a.strideRow
  inputInnerStride := request.a.strideCol
  weightInnerStride := request.b.strideRow
  weightColumnStride := request.b.strideCol
  outputRowStride := max request.b.cols 1
  outputColumnStride := 1

/-- Primitive storage/view ABI. Pointer origins include the actual Tensor
storage offsets. FP32 words are encoded in the source's floating carrier;
packed storage is unsigned8 and C preserves the complete signed32 word. -/
def encodedState (request : Request) (out : Nat) (memory : Memory Word32) : State Word32 :=
  { mem := fun r o =>
      if r = inputPort then fp32Cell (memory ((request.a.storage,.fp32),request.a.offset+o))
      else if r = weightPort then uint8Cell (packedByte (memory ((request.b.storage,.uint8),request.b.offset+o)))
      else if r = outputPort then int32Cell (memory ((out,.int32),o))
      else .mk .int 0
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 0
    undef := fun d _ _ => defaultValue d }

def DefinedInputs (request : Request) (out : Nat) (memory : Memory Word32) : Prop :=
  ∀ m, m < request.shape.m → ∀ k, k < request.shape.k →
    DefinedCast (memory (aAddress (hostBuffers request out) m k))

/-- Independent Tensor/backing-memory target. The same admitted FP32 words
and uint8 storage observations appear here; no source evaluator is referenced. -/
def tensorSignedSum (request : Request) (out : Nat) (memory : Memory Word32) (m n : Nat) : Int :=
  denseSignedSum request.shape.k
    (fun k => memory (aAddress (hostBuffers request out) m k))
    (fun q => memory (bAddress (hostBuffers request out) q n))
def tensorWordTarget (request : Request) (out : Nat) (memory : Memory Word32) (m n : Nat) : Word32 :=
  BitVec.ofInt 32 (tensorSignedSum request out memory m n)

theorem input_encoding (request : Request) (out : Nat) (memory : Memory Word32) (m k : Nat) :
    inputWords (layout request) (encodedState request out memory) m k =
      memory (aAddress (hostBuffers request out) m k) := by
  simp [inputWords,layout,encodedState,fp32Cell,aAddress,hostBuffers,Buffers.aRegion,Nat.add_assoc]

theorem packed_encoding (request : Request) (out : Nat) (memory : Memory Word32) (n q : Nat) :
    packedValues (layout request) (encodedState request out memory) n q =
      (packedByte (memory (bAddress (hostBuffers request out) q n))).toNat := by
  simp [packedValues,layout,encodedState,uint8Cell,bAddress,hostBuffers,Buffers.bRegion,
    inputPort,weightPort,Nat.add_assoc]

theorem output_encoding (request : Request) (out : Nat) (memory : Memory Word32) (o : Nat) :
    (encodedState request out memory).mem outputPort o = int32Cell (memory ((out,.int32),o)) := by
  simp [encodedState,inputPort,weightPort,outputPort]

theorem source_domains (request : Request) (out : Nat) (memory : Memory Word32)
    (defined : DefinedInputs request out memory) :
    FP32Domain (layout request) request.shape (encodedState request out memory) ∧
      UInt8Domain (layout request) request.shape (encodedState request out memory) := by
  constructor
  · intro m hm k hk
    rw [input_encoding]
    exact defined m hm k hk
  · intro n hn q hq
    rw [packed_encoding]
    exact (packedByte (memory (bAddress (hostBuffers request out) q n))).isLt

theorem repack (word : Word32) : packedByte (BitVec.ofNat 32 (packedByte word).toNat) = packedByte word := by
  apply BitVec.eq_of_toNat_eq
  have bound := (packedByte word).isLt
  change (packedByte word).toNat % 4294967296 % 256 = (packedByte word).toNat
  rw [Nat.mod_eq_of_lt (by omega : (packedByte word).toNat < 4294967296),Nat.mod_eq_of_lt bound]

theorem target_encoding (request : Request) (out : Nat) (memory : Memory Word32) (m n : Nat) :
    matrixWordTarget (layout request) request.shape (encodedState request out memory) m n =
      tensorWordTarget request out memory m n := by
  unfold matrixWordTarget tensorWordTarget tensorSignedSum denseWordTarget denseSignedSum
  congr 1
  apply Finset.sum_congr rfl
  intro f hf
  apply Finset.sum_congr rfl
  intro q hq
  rw [input_encoding]
  simp only [packedWords,packed_encoding,repack]

/-- Read the actual returned INT32 Tensor through its own storage identity,
storage offset, and strides in the common backing memory. -/
def tensorRead32 (tensor : Tensor) (memory : Memory Word32) (m n : Nat) : Word32 :=
  memory ((tensor.storage,.int32),tensor.offset+m*tensor.strideRow+n*tensor.strideCol)

/-- Scalar INT32 store decoding on the output port. All other typed storage
observations retain their original identities and values. The following
readback/frame lemmas prove the common-memory connection explicitly. -/
def decodeBacking (out : Nat) (before : Memory Word32) (after : Backing) : Memory Word32 :=
  fun loc => if loc.1 = (out,.int32) then BitVec.ofInt 32 ((after (outputPort,loc.2)).read .int)
    else before loc

def resultMemory (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15) : Memory Word32 :=
  decodeBacking out before (gridMemory (layout request) request.shape c (encodedState request out before))

theorem backing_output (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15)
    (m n : Nat) (hm : m < request.shape.m) (hn : n < request.shape.n) :
    resultMemory request out before c (outAddress request.shape (hostBuffers request out) m n) =
      tensorWordTarget request out before m n := by
  unfold resultMemory decodeBacking
  simp only [outAddress,hostBuffers,Buffers.outRegion,Prod.fst,Prod.snd,ite_true]
  erw [grid_values (layout request) request.shape c (encodedState request out before)
    (by exact ⟨rfl,rfl⟩) m n hm hn]
  rw [int32_read,BitVec.ofInt_toInt,target_encoding]

theorem tensor_readback (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15)
    (m n : Nat) (hm : m < request.shape.m) (hn : n < request.shape.n) :
    tensorRead32 (outputTensor request out) (resultMemory request out before c) m n =
      tensorWordTarget request out before m n := by
  convert backing_output request out before c m n hm hn using 1 <;>
    simp [tensorRead32,outputTensor,outAddress,Whole001.outputOffset,hostBuffers,Buffers.outRegion,Request.shape]

theorem backing_frame (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15)
    (loc : Address) (outside : loc.1 ≠ (out,.int32) ∨ request.shape.m*request.shape.n ≤ loc.2) :
    resultMemory request out before c loc = before loc := by
  unfold resultMemory decodeBacking
  by_cases hr : loc.1 = (out,.int32)
  · rw [if_pos hr]
    have ho : request.shape.m*request.shape.n ≤ loc.2 := by tauto
    rw [grid_frame (layout request) request.shape c (encodedState request out before)
      (by exact ⟨rfl,rfl⟩) (outputPort,loc.2) (Or.inr ho)]
    unfold flatten
    rw [output_encoding,int32_read,BitVec.ofInt_toInt]
    exact congrArg before (Prod.ext hr.symm rfl)
  · exact if_neg hr

theorem caller_frame (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15)
    (caller : Set Nat) (fresh : out ∉ caller) (loc : Address) (owned : loc.1.1 ∈ caller) :
    resultMemory request out before c loc = before loc := by
  apply backing_frame request out before c loc
  left
  intro he
  have hstorage : loc.1.1 = out := congrArg Prod.fst he
  rw [hstorage] at owned
  exact fresh owned

#axiomsClean input_encoding
#axiomsClean packed_encoding
#axiomsClean output_encoding
#axiomsClean source_domains
#axiomsClean repack
#axiomsClean target_encoding
#axiomsClean backing_output
#axiomsClean tensor_readback
#axiomsClean backing_frame
#axiomsClean caller_frame
#specNonCircular tensorSignedSum avoiding [encodedState, resultMemory, source, gridMemory]
#specNonCircular tensorWordTarget avoiding [encodedState, resultMemory, source, gridMemory]
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
