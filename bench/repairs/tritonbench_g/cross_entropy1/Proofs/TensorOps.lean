import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AggregateTrace

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton FP.Structural Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

/-- Host primitives have their own arithmetic ports: each call reads stored
operands and completes a separate operation, so contraction cannot cross calls. -/
structure TensorArithmetic (α : Type) where
  zero : α
  add : α → α → α
  square : α → α
  scale : ℝ → α → α
  sum : List α → α
  logsumexp : List α → α
  rankSum : List α → α

structure FloatArray {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (s : World α σ) (t : Tensor) (shape : List Nat) (values : Nat → α) : Prop where
  shapeEq : t.shape=shape
  dtype : t.dtype="torch.float32"
  live : s.live t.region
  values : ∀ i, i<t.numel → value obs s t i=values i

structure Produced {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (before after : World α σ) (t : Tensor) (shape : List Nat) (device : Nat) (values : Nat → α) : Prop where
  allocation : Allocation obs t shape "torch.float32" device before after
  dense : t.Dense
  values : ∀ i, i<t.numel → value obs after t i=values i

theorem Produced.array {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {before after : World α σ} {t : Tensor} {shape : List Nat} {device : Nat} {v : Nat → α}
    (h : Produced obs before after t shape device v) : FloatArray obs after t shape v :=
  ⟨h.allocation.shapeEq,h.allocation.dtypeEq,h.allocation.live,h.values⟩

theorem FloatArray.preserve {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {before after : World α σ} {t : Tensor} {shape : List Nat} {v : Nat → α}
    (h : FloatArray obs before t shape v) (frame : Preserves obs before after) :
    FloatArray obs after t shape v :=
  ⟨h.shapeEq,h.dtype,frame.live _ h.live,fun i hi => (frame.floatValue t h.live i).trans (h.values i hi)⟩

/-- An ordinary in-place tensor operation changes only coordinates of its
destination. Integer views may share its bytes; they are preserved only when
their storage region differs. Physical-byte framing is supplied separately. -/
structure Mutation {α σ : Type} (obs : Observations α σ) (before after : World α σ) (dst : Tensor) : Prop where
  live : ∀ r, before.live r → after.live r
  memory : ∀ r i, before.live r → (r≠dst.region ∨ ∀ j, j<dst.numel → i≠dst.index j) → after.memory r i=before.memory r i
  integers : ∀ t i, before.live t.region → t.region≠dst.region → obs.integer after t i=obs.integer before t i

theorem Mutation.floatValue {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {before after : World α σ} {dst : Tensor} (h : Mutation obs before after dst)
    (t : Tensor) (live : before.live t.region) (separate : t.region≠dst.region) (i : Nat) : value obs after t i=value obs before t i := by
  unfold VeriTile.Bench.Repairs.CrossEntropy1.Host.value
  rw [h.memory _ _ live (Or.inl separate)]

theorem FloatArray.mutate {α σ : Type} [Inhabited α] {obs : Observations α σ}
    {before after : World α σ} {t dst : Tensor} {shape : List Nat} {v : Nat → α}
    (h : FloatArray obs before t shape v) (frame : Mutation obs before after dst)
    (separate : t.region≠dst.region) : FloatArray obs after t shape v :=
  ⟨h.shapeEq,h.dtype,frame.live _ h.live,fun i hi => (frame.floatValue t h.live separate i).trans (h.values i hi)⟩

def columnValues {α σ : Type} [Inhabited α] (obs : Observations α σ) (s : World α σ)
    (t : Tensor) (blocks rows row : Nat) : List α :=
  (List.range blocks).map (fun block => value obs s t (block*rows+row))

/-- Each premise and result belongs to one ordinary tensor-library call.
None mentions cross entropy, a JIT, a grid or a whole host result. -/
structure TensorLibrary {α σ : Type} [Inhabited α] (obs : Observations α σ)
    (A : TensorArithmetic α) (rt : Runtime α σ) : Prop where
  logsumexp : ∀ input blocks rows before output after, before.live input.region →
    input.shape=[blocks,rows] → input.dtype="torch.float32" →
    rt.logsumexp0 input before=.ok output after →
      Produced obs before after output [rows] input.device
        (fun row => A.logsumexp (columnValues obs before input blocks rows row))
  sum : ∀ input blocks rows before output after, before.live input.region →
    input.shape=[blocks,rows] → input.dtype="torch.float32" →
    rt.sum0 input before=.ok output after →
      Produced obs before after output [rows] input.device
        (fun row => A.sum (columnValues obs before input blocks rows row))
  square : ∀ input before output after, before.live input.region → input.dtype="torch.float32" →
    rt.square input before=.ok output after →
      Produced obs before after output input.shape input.device (fun i => A.square (value obs before input i))
  scalarMul : ∀ scale input before output after, before.live input.region → input.dtype="torch.float32" →
    rt.scalarMul scale input before=.ok output after →
      Produced obs before after output input.shape input.device (fun i => A.scale scale (value obs before input i))
  add : ∀ dst src before output after, before.live dst.region → before.live src.region →
    dst.dtype="torch.float32" → src.dtype="torch.float32" → dst.shape=src.shape →
    rt.addInplace dst src before=.ok output after → output=dst ∧ Mutation obs before after dst ∧
      (∀ i, i<dst.numel → value obs after dst i=A.add (value obs before dst i) (value obs before src i))
  equalInt : ∀ input target before output after, before.live input.region → input.dtype="torch.int64" →
    rt.equalInt input target before=.ok output after →
      Allocation obs output input.shape "torch.bool" input.device before after ∧ output.Dense ∧
      (∀ i, i < input.numel → (after.memory output.region i).read .bool=decide (obs.integer before input i=target))
  maskedFill : ∀ dst mask before output after, before.live dst.region → before.live mask.region →
    dst.dtype="torch.float32" → mask.dtype="torch.bool" → dst.shape=mask.shape →
    rt.maskedFillZero dst mask before=.ok output after → output=dst ∧ Mutation obs before after dst ∧
      (∀ i, i<dst.numel → value obs after dst i=
        if (before.memory mask.region (mask.index i)).read .bool then A.zero else value obs before dst i)

/-- Registry membership binds object identity to all tensor-view metadata.
The registry covers the finite lifetime of this forward/backward invocation;
new objects may be fresh even when an allocator later reuses an address. -/
structure ObjectRegistry {α σ : Type} (rt : Runtime α σ) where
  valid : Tensor → Prop
  identity : ∀ a b, valid a → valid b → a.objectId=b.objectId → a=b
  resolve : ∀ input before output after, rt.resolveNeg input before=.ok output after →
    valid output ∧ (output=input ∨ ¬before.live output.region)
  contiguous : ∀ input before output after, rt.contiguous input before=.ok output after →
    valid output ∧ (output=input ∨ ¬before.live output.region)
  clone : ∀ input before output after, rt.cloneContiguous input before=.ok output after → valid output
  empty : ∀ shape dtype device before output after, rt.empty shape dtype device before=.ok output after → valid output
  emptyLike : ∀ input packed before output after, rt.emptyLike input packed before=.ok output after → valid output
  logsumexp : ∀ input before output after, rt.logsumexp0 input before=.ok output after → valid output
  sum : ∀ input before output after, rt.sum0 input before=.ok output after → valid output
  square : ∀ input before output after, rt.square input before=.ok output after → valid output
  scalarMul : ∀ scale input before output after, rt.scalarMul scale input before=.ok output after → valid output
  equalInt : ∀ input target before output after, rt.equalInt input target before=.ok output after → valid output

#axiomsClean Produced.array
#axiomsClean FloatArray.preserve
#axiomsClean FloatArray.mutate
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
