import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Bounds
import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Numeric
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section

/-- Independent mathematical product on original FP32 caller words and packed
uint8 B, with each FP32 entry truncated toward zero into signed8. -/
def quantizedDenseTarget (r : Request) (out : Nat) (before : Memory Word) (m n : Nat) : Int :=
  denseTarget referenceCast packedByte r.shape.k
    (fun k => before (aAddress (hostBuffers r out) m k))
    (fun k => before (bAddress (hostBuffers r out) k n))

/-- Independent finite signed32 result, preserving selected ISA overflow. -/
def finiteTarget (r : Request) (out : Nat) (c : Fin 15) (before : Memory Word) (m n : Nat) : Word :=
  wordReference (selectedInstruction c) r.shape.k
    (fun k => before (aAddress (hostBuffers r out) m k))
    (fun k => before (bAddress (hostBuffers r out) k n))

structure WholeDomain (r : Request) (b : Buffers) (before : Memory Word)
    (trials : List (Fin 15)) (selected : Fin 15) where
  host : r.Domain
  scalarBounds : SourceBounds r.shape b
  /-- Actual allocation-base metadata supplied by the execution binding. -/
  pointers : PointerBounds r.shape b
  defined : InputDefined r.shape b before
  selectedLegal : r.shape.legal selected
  trialsLegal : ∀ c∈trials,r.shape.legal c

/-- Keep the caller's original words while exposing an unbounded mathematical
output cell. This is the integer interpretation of the source arithmetic,
not an assertion that a finite signed32 device output never overflows. -/
abbrev MathCell := Word × Int
def mathLift (m : Memory Word) : Memory MathCell := fun p => (m p,0)
def integerArithmetic : Arithmetic MathCell Int :=
  mathematical (fun x => referenceCast x.1) (fun x => packedByte x.1) (fun z => (0,z))

def Allocation.map (h : Allocation r caller (before : Memory α)) (f : α → γ) :
    Allocation r caller (fun p => f (before p)) where
  output := h.output
  fresh := h.fresh
  aCaller := h.aCaller
  bCaller := h.bCaller
  initial := fun p => f (h.initial p)
  preserves := fun p hp => congrArg f (h.preserves p hp)

def Allocation.math (h : Allocation r caller (before : Memory Word)) :
    Allocation r caller (mathLift before) := h.map (fun w => (w,0))

/-- Host-visible return metadata, independent of output values. -/
def ReturnContract (r : Request) (caller : Set Nat) (out : Tensor) : Prop :=
  out.rows=r.a.rows ∧ out.cols=r.b.cols ∧ out.dtype=.int32 ∧
  out.device=r.a.device ∧ out.offset=0 ∧ out.strideRow=max r.b.cols 1 ∧
  out.strideCol=1 ∧ out.resolved=true ∧ out.cuda=r.a.cuda ∧
  out.capacity=r.a.rows*r.b.cols ∧ out.contiguous ∧ out.storage∉caller

theorem returned_contract (h : Allocation r caller (before : Memory α)) :
    ReturnContract r caller h.returned := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,
    Or.inr ⟨Or.inr rfl,Or.inr rfl⟩,h.fresh⟩

#axiomsClean returned_contract
#specNonCircular quantizedDenseTarget avoiding [run,sourceAccumulation,accumulation]
#specNonCircular finiteTarget avoiding [run,sourceAccumulation,accumulation]
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
