import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Library
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Mask

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step Observations Preserves Mutation FloatArray Produced Allocation IntArray value)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

structure IntegerArray {α σ : Type} (obs : Observations α σ) (dtype : String)
    (s : World α σ) (t : Tensor) (shape : List Nat) (v : Nat → Int) : Prop where
  shapeEq : t.shape=shape
  dtypeEq : t.dtype=dtype
  live : s.live t.region
  values : ∀ i, i < t.numel → obs.integer s t i=v i

theorem IntegerArray.preserve {α σ : Type} {obs : Observations α σ} {dtype : String}
    {a b : World α σ} {t : Tensor} {shape : List Nat} {v : Nat → Int}
    (array : IntegerArray obs dtype a t shape v) (frame : Preserves obs a b) : IntegerArray obs dtype b t shape v :=
  ⟨array.shapeEq,array.dtypeEq,frame.live _ array.live,fun i hi => (frame.integers t i array.live).trans (array.values i hi)⟩

theorem IntegerArray.mutate {α σ : Type} {obs : Observations α σ} {dtype : String}
    {a b : World α σ} {t dst : Tensor} {shape : List Nat} {v : Nat → Int}
    (array : IntegerArray obs dtype a t shape v) (frame : Mutation obs a b dst) (separate : t.region≠dst.region) :
    IntegerArray obs dtype b t shape v :=
  ⟨array.shapeEq,array.dtypeEq,frame.live _ array.live,
   fun i hi => (frame.integers t i array.live separate).trans (array.values i hi)⟩

theorem add_arrays {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (left right output : Tensor) (before after : World α σ)
    (shape : List Nat) (x y : Nat → α)
    (lhs : FloatArray obs before left shape x) (rhs : FloatArray obs before right shape y)
    (call : rt.addInplace left right before=.ok output after) :
    output=left ∧ Mutation obs before after left ∧
      FloatArray obs after output shape (fun i => A.add (x i) (y i)) := by
  have op := lib.operations.add left right before output after lhs.live rhs.live lhs.dtype rhs.dtype
    (lhs.shapeEq.trans rhs.shapeEq.symm) call
  refine ⟨op.1,op.2.1,?_⟩
  rw [op.1]
  refine ⟨lhs.shapeEq,lhs.dtype,op.2.1.live _ lhs.live,?_⟩
  intro i hi
  rw [op.2.2 i hi,lhs.values i hi,rhs.values i (by simpa only [Tensor.numel,lhs.shapeEq,rhs.shapeEq] using hi)]

theorem scale_array {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (scale : ℝ) (input output : Tensor) (before after : World α σ)
    (shape : List Nat) (x : Nat → α) (array : FloatArray obs before input shape x)
    (call : rt.scalarMul scale input before=.ok output after) :
    Produced obs before after output shape input.device (fun i => A.scale scale (x i)) := by
  have op := lib.operations.scalarMul scale input before output after array.live array.dtype call
  refine ⟨?_,op.dense,?_⟩
  · simpa only [array.shapeEq] using op.allocation
  · intro i hi
    rw [op.values i hi,array.values i (by simpa only [Tensor.numel,op.allocation.shapeEq] using hi)]

theorem multiply_arrays {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (left right output : Tensor) (before after : World α σ)
    (shape : List Nat) (x y : Nat → α)
    (lhs : FloatArray obs before left shape x) (rhs : FloatArray obs before right shape y)
    (call : rt.multiply left right before=.ok output after) :
    Produced obs before after output shape left.device (fun i => A.multiply (x i) (y i)) := by
  have op := lib.multiply left right before output after lhs.live rhs.live lhs.dtype rhs.dtype
    (lhs.shapeEq.trans rhs.shapeEq.symm) call
  refine ⟨?_,op.dense,?_⟩
  · simpa only [lhs.shapeEq] using op.allocation
  · intro i hi
    rw [op.values i hi,lhs.values i (by simpa only [Tensor.numel,op.allocation.shapeEq] using hi),
      rhs.values i (by simpa only [Tensor.numel,op.allocation.shapeEq,lhs.shapeEq,rhs.shapeEq] using hi)]

theorem widen_array {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (input output : Tensor) (before after : World α σ)
    (shape : List Nat) (x : Nat → Int) (array : IntegerArray obs "torch.int32" before input shape x)
    (call : rt.toInt64 input before=.ok output after) :
    Allocation obs output shape "torch.int64" input.device before after ∧ output.Dense ∧
      IntArray obs after output shape x := by
  have op := lib.castInt64 input before output after array.live array.dtypeEq call
  refine ⟨?_,op.2.1,?_⟩
  · simpa only [array.shapeEq] using op.1
  · refine ⟨op.1.shapeEq.trans array.shapeEq,op.1.dtypeEq,op.1.live,?_⟩
    intro i hi
    have bound : i < input.numel := by simpa only [Tensor.numel,op.1.shapeEq] using hi
    exact (op.2.2 i bound).trans (array.values i bound)

structure BooleanArray {α σ : Type} (s : World α σ) (t : Tensor) (shape : List Nat) (v : Nat → Bool) : Prop where
  shapeEq : t.shape=shape
  dtypeEq : t.dtype="torch.bool"
  live : s.live t.region
  values : ∀ i, i < t.numel → (s.memory t.region (t.index i)).read .bool=v i

theorem BooleanArray.preserve {α σ : Type} {obs : Observations α σ}
    {a b : World α σ} {t : Tensor} {shape : List Nat} {v : Nat → Bool}
    (array : BooleanArray a t shape v) (frame : Preserves obs a b) : BooleanArray b t shape v :=
  ⟨array.shapeEq,array.dtypeEq,frame.live _ array.live,fun i hi => by rw [frame.memory _ _ array.live];exact array.values i hi⟩

theorem BooleanArray.mutate {α σ : Type} {obs : Observations α σ}
    {a b : World α σ} {t dst : Tensor} {shape : List Nat} {v : Nat → Bool}
    (array : BooleanArray a t shape v) (frame : Mutation obs a b dst) (separate : t.region≠dst.region) : BooleanArray b t shape v :=
  ⟨array.shapeEq,array.dtypeEq,frame.live _ array.live,
    fun i hi => by rw [frame.memory _ _ array.live (Or.inl separate)];exact array.values i hi⟩

theorem compare_array {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (input output : Tensor) (target : Int) (before after : World α σ)
    (shape : List Nat) (x : Nat → Int) (array : IntArray obs before input shape x)
    (call : rt.equalInt input target before=.ok output after) :
    Allocation obs output shape "torch.bool" input.device before after ∧ output.Dense ∧
      BooleanArray after output shape (fun i => decide (x i=target)) := by
  have op := lib.operations.equalInt input target before output after array.live array.dtype call
  refine ⟨?_,op.2.1,?_⟩
  · simpa only [array.shapeEq] using op.1
  · refine ⟨op.1.shapeEq.trans array.shapeEq,op.1.dtypeEq,op.1.live,?_⟩
    intro i hi
    have bound : i < input.numel := by simpa only [Tensor.numel,op.1.shapeEq] using hi
    rw [op.2.1.2 i hi,op.2.2 i bound,array.values i bound]

theorem mask_array {α σ : Type} [Inhabited α] (obs : Observations α σ) (A : Arithmetic α)
    (rt : Runtime α σ) (lib : Library obs A rt) (input mask output : Tensor) (before after : World α σ)
    (shape : List Nat) (x : Nat → α) (flags : Nat → Bool)
    (array : FloatArray obs before input shape x) (booleans : BooleanArray before mask shape flags)
    (call : rt.maskedFillZero input mask before=.ok output after) :
    output=input ∧ Mutation obs before after input ∧
      FloatArray obs after output shape (fun i => if flags i then A.zero else x i) := by
  have op := lib.operations.maskedFill input mask before output after array.live booleans.live array.dtype booleans.dtypeEq
    (array.shapeEq.trans booleans.shapeEq.symm) call
  refine ⟨op.1,op.2.1,?_⟩
  rw [op.1]
  refine ⟨array.shapeEq,array.dtype,op.2.1.live _ array.live,?_⟩
  intro i hi
  rw [op.2.2 i hi,booleans.values i (by simpa only [Tensor.numel,array.shapeEq,booleans.shapeEq] using hi),array.values i hi]

#axiomsClean add_arrays
#axiomsClean multiply_arrays
#axiomsClean widen_array
#axiomsClean mask_array
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
