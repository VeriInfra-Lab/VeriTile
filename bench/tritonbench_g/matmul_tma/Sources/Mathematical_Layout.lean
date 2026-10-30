import bench.tritonbench_g.matmul_tma.Sources.Mathematical_Kernels

/-! Source wrapper metadata, allocation bounds and launch request. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTma.Host
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Shape where
  rows : Nat
  columns : Nat
  inner : Nat
  deriving DecidableEq, Repr

structure Flags where
  transA : Bool
  transB : Bool
  half : Bool
  deriving DecidableEq, Repr

def allFlags : List Flags :=
  [⟨Bool.false,Bool.false,Bool.false⟩, ⟨Bool.true,Bool.false,Bool.false⟩,
   ⟨Bool.false,Bool.true,Bool.false⟩, ⟨Bool.true,Bool.true,Bool.false⟩,
   ⟨Bool.false,Bool.false,Bool.true⟩, ⟨Bool.true,Bool.false,Bool.true⟩,
   ⟨Bool.false,Bool.true,Bool.true⟩, ⟨Bool.true,Bool.true,Bool.true⟩]

theorem all_flags (f : Flags) : f ∈ allFlags := by
  rcases f with ⟨a,b,c⟩; cases a <;> cases b <;> cases c <;> decide

structure Config where
  numCtas : Nat
  numWarps : Nat
  deriving DecidableEq, Repr

/-- Allocation identities. `replacement` is used only on OUTPUT_F16=Bool.true. -/
structure Regions where
  a : RegionName
  b : RegionName
  first : RegionName
  replacement : RegionName

def outputRegion (r : Regions) (f : Flags) : RegionName :=
  if f.half then r.replacement else r.first

structure Tensor where
  region : RegionName
  dtype : FloatDType
  rows : Nat
  columns : Nat
  strideRow : Nat
  strideColumn : Nat

def contiguous (r : RegionName) (d : FloatDType) (m n : Nat) : Tensor :=
  ⟨r,d,m,n,n,1⟩
def transpose (t : Tensor) : Tensor :=
  ⟨t.region, t.dtype, t.columns, t.rows, t.strideColumn, t.strideRow⟩

def inputA (h : Shape) (f : Flags) (r : Regions) : Tensor :=
  if f.transA then transpose (contiguous r.a .fp16 h.inner h.rows)
  else contiguous r.a .fp16 h.rows h.inner

def inputB (h : Shape) (f : Flags) (r : Regions) : Tensor :=
  if f.transB then transpose (contiguous r.b .fp16 h.columns h.inner)
  else contiguous r.b .fp16 h.inner h.columns

def firstOutput (h : Shape) (r : Regions) : Tensor :=
  contiguous r.first .fp32 h.rows h.columns

def returnedTensor (h : Shape) (f : Flags) (r : Regions) : Tensor :=
  if f.half then contiguous r.replacement .fp16 h.rows h.columns
  else firstOutput h r

/-- All runtime strides are obtained from the views, as in the Python call. -/
def parameters (h : Shape) (f : Flags) (r : Regions) : Parameters where
  a := (inputA h f r).region
  b := (inputB h f r).region
  c := (returnedTensor h f r).region
  rows := h.rows
  columns := h.columns
  inner := h.inner
  aRow := (inputA h f r).strideRow
  aInner := (inputA h f r).strideColumn
  bInner := (inputB h f r).strideRow
  bColumn := (inputB h f r).strideColumn
  cRow := (returnedTensor h f r).strideRow
  cColumn := (returnedTensor h f r).strideColumn
  blockRows := h.rows
  blockColumns := h.columns
  blockInner := h.inner

@[simp] theorem inputA_region (h : Shape) (f : Flags) (r : Regions) :
    (inputA h f r).region = r.a := by
  cases hf : f.transA <;> simp [inputA,hf,contiguous,transpose]
@[simp] theorem inputB_region (h : Shape) (f : Flags) (r : Regions) :
    (inputB h f r).region = r.b := by
  cases hf : f.transB <;> simp [inputB,hf,contiguous,transpose]
@[simp] theorem returned_region (h : Shape) (f : Flags) (r : Regions) :
    (returnedTensor h f r).region = outputRegion r f := by
  cases hf : f.half <;> simp [returnedTensor,outputRegion,outputDType,hf,firstOutput,contiguous]
@[simp] theorem returned_dtype (h : Shape) (f : Flags) (r : Regions) :
    (returnedTensor h f r).dtype = outputDType f.half := by
  cases hf : f.half <;> simp [returnedTensor,outputRegion,outputDType,hf,firstOutput,contiguous]
@[simp] theorem returned_shape (h : Shape) (f : Flags) (r : Regions) :
    (returnedTensor h f r).rows = h.rows ∧
    (returnedTensor h f r).columns = h.columns ∧
    (returnedTensor h f r).strideRow = h.columns ∧
    (returnedTensor h f r).strideColumn = 1 := by
  cases hf : f.half <;> simp [returnedTensor,hf,firstOutput,contiguous]

theorem input_metadata (h : Shape) (f : Flags) (r : Regions) :
    (inputA h f r).dtype = .fp16 ∧ (inputB h f r).dtype = .fp16 ∧
    (inputA h f r).rows = h.rows ∧ (inputA h f r).columns = h.inner ∧
    (inputB h f r).rows = h.inner ∧ (inputB h f r).columns = h.columns := by
  cases ha : f.transA <;> cases hb : f.transB <;> simp [inputA,inputB,ha,hb,contiguous,transpose]

theorem rowMajor_bound {m n : Nat} (i : Fin m) (j : Fin n) :
    i.val * n + j.val < m * n := by
  have hm := Nat.mul_le_mul_right n (Nat.succ_le_of_lt i.isLt)
  simp only [Nat.succ_mul] at hm
  omega

theorem a_bound (h : Shape) (f : Flags) (r : Regions)
    (i : AIndex (parameters h f r)) :
    aOffset (parameters h f r) i < h.rows * h.inner := by
  have hn := rowMajor_bound i.1 i.2.1
  have ht := rowMajor_bound i.2.1 i.1
  cases ha : f.transA
  · simpa [aOffset, parameters, inputA, ha, transpose, contiguous] using hn
  · simpa [aOffset, parameters, inputA, ha, transpose, contiguous, Nat.mul_comm, Nat.add_comm] using ht

theorem b_bound (h : Shape) (f : Flags) (r : Regions)
    (i : BIndex (parameters h f r)) :
    bOffset (parameters h f r) i < h.inner * h.columns := by
  have hn := rowMajor_bound i.1 i.2.1
  have ht := rowMajor_bound i.2.1 i.1
  cases hb : f.transB
  · simpa [bOffset, parameters, inputB, hb, transpose, contiguous] using hn
  · simpa [bOffset, parameters, inputB, hb, transpose, contiguous, Nat.mul_comm, Nat.add_comm] using ht

theorem c_offset (h : Shape) (f : Flags) (r : Regions)
    (i : CIndex (parameters h f r)) :
    FPValue.cOffset (parameters h f r) i = i.1.val * h.columns + i.2.1.val := by
  cases hf : f.half <;> simp [FPValue.cOffset, parameters, returnedTensor, hf,
    firstOutput, contiguous]

theorem c_bound (h : Shape) (f : Flags) (r : Regions)
    (i : CIndex (parameters h f r)) :
    FPValue.cOffset (parameters h f r) i < h.rows * h.columns := by
  rw [c_offset]; exact rowMajor_bound i.1 i.2.1

theorem c_injective (h : Shape) (f : Flags) (r : Regions) :
    Function.Injective (FPValue.cOffset (parameters h f r)) := by
  intro i j hij
  apply rowMajor2D_inj 0 h.columns (Nat.le_refl _)
  simpa only [c_offset, Nat.zero_add] using hij

/-- Capacity/freshness belong to the ordinary allocation boundary. They do not
assert a launch, a numerical result, or successful source execution. -/
structure AllocationBinding (h : Shape) (f : Flags) (r : Regions) (a : FlatAlloc) : Prop where
  disjoint : a.Disjoint
  a_mem : r.a ∈ a.regions
  b_mem : r.b ∈ a.regions
  first_mem : r.first ∈ a.regions
  replacement_mem : f.half = Bool.true → r.replacement ∈ a.regions
  a_extent : h.rows * h.inner ≤ a.extent r.a
  b_extent : h.inner * h.columns ≤ a.extent r.b
  first_extent : h.rows * h.columns ≤ a.extent r.first
  replacement_extent : f.half = Bool.true → h.rows * h.columns ≤ a.extent r.replacement
  inputs_distinct : r.a ≠ r.b
  first_fresh : r.first ≠ r.a ∧ r.first ≠ r.b
  replacement_fresh : f.half = Bool.true →
    r.replacement ≠ r.a ∧ r.replacement ≠ r.b ∧ r.replacement ≠ r.first

theorem output_capacity (h : Shape) (f : Flags) (r : Regions) (a : FlatAlloc)
    (binding : AllocationBinding h f r a) :
    h.rows * h.columns ≤ a.extent (outputRegion r f) := by
  cases hf : f.half
  · simpa [outputRegion,hf] using binding.first_extent
  · simpa [outputRegion,hf] using binding.replacement_extent hf

theorem all_accesses_bounded (h : Shape) (f : Flags) (r : Regions) (a : FlatAlloc)
    (binding : AllocationBinding h f r a) :
    (∀ i, aOffset (parameters h f r) i < a.extent (parameters h f r).a) ∧
    (∀ i, bOffset (parameters h f r) i < a.extent (parameters h f r).b) ∧
    (∀ i, FPValue.cOffset (parameters h f r) i < a.extent (parameters h f r).c) := by
  refine ⟨fun i => ?_, fun i => ?_, fun i => ?_⟩
  · simpa [parameters] using lt_of_lt_of_le (a_bound h f r i) binding.a_extent
  · simpa [parameters] using lt_of_lt_of_le (b_bound h f r i) binding.b_extent
  · simpa [parameters] using lt_of_lt_of_le (c_bound h f r i) (output_capacity h f r a binding)

/-- Exact compiler/runtime request. Block-pointer orders are source metadata,
not a predicate requiring the fastest-stride axis to match the order tuple. -/
structure Request where
  parameters : Parameters
  config : Config
  output : FloatDType
  grid : Nat × Nat
  aOrder : List Nat
  bOrder : List Nat
  cOrder : List Nat
  input : FloatDType
  accumulator : FloatDType
  precision : Option ComputeDType
  dotOptions : DotOptions

def request (h : Shape) (f : Flags) (r : Regions) (cfg : Config) : Request :=
  ⟨parameters h f r, cfg, outputDType f.half, (1,1), [1,0], [0,1], [1,0],
    .fp16, .fp32, some .fp32, {}⟩

/-- Fallible native setup only: compilation/configuration, pointer packing,
context and launch admission. The function has no state or tensor data and
cannot supply a kernel result. Concrete compile evidence is kept separately. -/
abbrev Admission := Request → Except String Unit

#axiomsClean all_accesses_bounded
#axiomsClean c_injective
#axiomsClean all_flags
end
end VeriTile.Bench.TritonBenchG.MatmulTma.Host
