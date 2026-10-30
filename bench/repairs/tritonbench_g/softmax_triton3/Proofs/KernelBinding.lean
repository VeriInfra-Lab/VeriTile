import bench.repairs.tritonbench_g.softmax_triton3.Proofs.Addresses
import bench.repairs.tritonbench_g.softmax_triton3.Proofs.KernelSource
namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
noncomputable section

/-- One physical-address reader binds every actual storage allocation. Input
and mask aliases therefore observe the same words at the same address. -/
def InputMemory {S C V} (a : Access S C V) (read : Nat → V) (x : Object) (s : S) : Prop :=
  ∀ i, i<x.tensor.storage.elements → a.read s x.region i=read (x.tensor.storage.address i)

/-- Independent values at original Tensor coordinates, before the row runs. -/
def logicalValue {V} (neg : V → V) (read : Nat → V) (x : Object) (r c : Nat) : V :=
  let value := read (x.tensor.storage.address (x.tensor.physicalIndex (r*cols x.metadata+c)))
  if x.metadata.isNeg then neg value else value

def logicalMask {V} (neg : V → V) (read : Nat → V) (mask : Option Object) (r : Nat) :
    Option (Nat → V) := mask.map (fun m => logicalValue neg read m r)

/-- The native pointer, logical Tensor element, and the source load coincide.
This lemma is used to supply the source execution's actual input values. -/
theorem bound_load {S C V} (a : Access S C V) (neg : V → V) (read : Nat → V)
    (x : Object) (h : TensorContract x.metadata) (s : S) (binding : InputMemory a read x s)
    (r c : Nat) (hr : r<rows x.metadata) (hc : c<cols x.metadata) :
    a.read s x.region ((descriptor x).index r c)=
      read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat ∧
    read (Native.sequentialSourcePointer (buffer x) (descriptor x) r c).toNat=
      logicalValue neg read x r c := by
  have hi : r*cols x.metadata+c<x.metadata.numel := by
    rw [numel_eq _ h.rank]
    exact VeriTile.Bench.Repairs.SquareMatrix.flat_lt _ _ _ _ hr hc
  have bounds := x.laws.inBounds _ hi
  have pointer := (pointer_bounds x h r c hr hc).1
  constructor
  · rw [descriptor_read a neg s x h r c hr hc,pointer]
    simp only [objectValue,h.noNeg,Bool.false_eq_true,↓reduceIte]
    exact binding _ bounds
  · rw [pointer]
    simp [logicalValue,h.noNeg]

/-- A successful single row invocation at actual Tensor windows. Output is a
caller-supplied contiguous FP32 Tensor of the same shape; freshness/allocation
are irrelevant to this kernel theorem. Every cell outside this row is framed.
The admitted input and mask may overlap. -/
theorem checked_row {S C V} (a : Access S C V) (memory : Memory a) (neg : V → V)
    (evaluate : Evaluator S) (formula : Formula V) (source : Source a evaluate formula)
    (terminates : ∀ out inp mask rows n B p,0<B → ∀ s,∃ t,evaluate out inp mask rows n B p s=some t)
    (x y : Object) (mask : Option Object) (dim : Int)
    (h : InputContract x.metadata (mask.map (·.metadata)) dim)
    (output : TensorContract y.metadata) (shape : y.metadata.shape=x.metadata.shape)
    (read : Nat → V) (s : S) (input : InputMemory a read x s)
    (masks : ∀ m,mask=some m → InputMemory a read m s)
    (p : Nat) (hp : p<rows x.metadata) :
    ∃ t,physicalEvaluate memory evaluate x y mask (blockSize (cols x.metadata)) p s=some t ∧
      (∀ c,c<cols x.metadata →
        a.read t y.region (y.tensor.physicalIndex (p*cols y.metadata+c))=
          formula (cols x.metadata) (blockSize (cols x.metadata))
            (logicalValue neg read x p) (logicalMask neg read mask p) c) ∧
      FrameRows a (descriptor y) [p] s t := by
  let B := blockSize (cols x.metadata)
  obtain ⟨u,run⟩ := terminates "out" "inp" (mask.map (fun _ => "mask"))
    (rows x.metadata) (cols x.metadata) B p (block_bounds _).1 (rebase memory x y mask s)
  let t := project memory y p s u
  have execute : physicalEvaluate memory evaluate x y mask B p s=some t := by
    simp [physicalEvaluate,run,t]
  have law := physical_row a memory evaluate formula source x y mask shape B p
    (block_bounds _).1 (block_bounds _).2 hp s t (logicalValue neg read x p)
    (logicalMask neg read mask p) (by
      intro c hc
      have load := bound_load a neg read x h.input s input p c hp hc
      exact load.1.trans load.2)
    (by
      intro m hm c hc
      have admitted := h.mask m.metadata (by simp [hm])
      have load := bound_load a neg read m admitted.1 s (masks m hm) p c
        (by rwa [admitted.2.2.2]) (by rwa [admitted.2.2.1])
      simpa [logicalMask,hm,descriptor,Tensor.index,admitted.2.2.1] using load.1.trans load.2)
    (by cases mask <;> rfl) execute
  refine ⟨t,execute,?_,law.2⟩
  intro c hc
  have yc : cols y.metadata=cols x.metadata := by simp [cols,shape]
  have yr : rows y.metadata=rows x.metadata := by simp [rows,shape]
  have value := law.1 c hc
  rw [descriptor_read a neg t y output p c (by rwa [yr]) (by rwa [yc])] at value
  simpa [objectValue,output.noNeg] using value

#axiomsClean bound_load
#axiomsClean checked_row
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Contiguous
