import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Host
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Native

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (coordinateOffset)
noncomputable section

structure Preserves (before after : State) : Prop where
  live : ∀ b,b∈before.live → b∈after.live
  frame : ∀ b,b∈before.live → Frame before.heap after.heap b

theorem Preserves.refl (s : State) : Preserves s s :=
  ⟨fun _ h => h,fun _ _ _ _ => rfl⟩

theorem Preserves.trans {a b c : State} (ab : Preserves a b) (bc : Preserves b c) : Preserves a c :=
  ⟨fun r h => bc.live r (ab.live r h),fun r h => (ab.frame r h).trans (bc.frame r (ab.live r h))⟩

structure Dense (t : Tensor) : Prop where
  normal : t.negative=Bool.false
  index : ∀ i,i<t.numel → t.index i=t.offset+i
  nonnull : 0<t.numel → 0<t.backing.base

/-- The empty adapter has no observable address. This retains legal empty
views with an arbitrary unused storage offset instead of restricting them. -/
def denseView (t : Tensor) (d : Dense t) : View where
  backing := t.backing
  offset := if t.numel=0 then 0 else t.offset
  words := t.numel
  capacity := by
    by_cases h : t.numel=0
    · simp [h]
    · have hi : t.numel-1<t.numel := by omega
      have cap := t.capacity (t.numel-1) hi
      have idx := d.index (t.numel-1) hi
      simp only [Tensor.index] at idx
      simp only [h,if_false]
      omega

theorem dense_address (t : Tensor) (d : Dense t) (i : Nat) (hi : i<t.numel) :
    (denseView t d).address i=t.address i := by
  have nonempty : t.numel≠0 := by omega
  simp only [denseView,nonempty,if_false,View.address,View.pointer,Tensor.address,d.index i hi]
  omega

theorem dense_value {Word : Type} (codec : Codec Word) (neg : Word→Word) (m : Heap)
    (t : Tensor) (d : Dense t) (i : Nat) (hi : i<t.numel) :
    load codec m ((denseView t d).address i)=value codec neg m t i := by
  rw [dense_address t d i hi]
  simp [value,d.normal]

structure Materialized {Word : Type} (codec : Codec Word) (neg : Word→Word)
    (input output : Tensor) (before after : State) : Prop where
  shape : output.shape=input.shape
  device : output.device=input.device
  preserved : Preserves before after
  live : output.backing∈after.live
  values : ∀ i,i < input.numel → value codec neg after.heap output i=value codec neg before.heap input i

structure Allocated (shape : List Nat) (device : Nat) (output : Tensor) (before after : State) : Prop where
  shape : output.shape=shape
  device : output.device=device
  dense : Dense output
  preserved : Preserves before after
  live : output.backing∈after.live
  fresh : ∀ b,b∈before.live → output.backing.disjoint b

structure Copied {Word : Type} (codec : Codec Word) (neg : Word→Word)
    (destination source returned : Tensor) (before after : State) : Prop where
  identity : returned=destination
  live : after.live=before.live
  values : ∀ i,i<destination.numel →
    value codec neg after.heap destination i=value codec neg before.heap source i
  frame : ∀ b,b∈before.live → ∀ p,b.contains p → ¬Written destination p → after.heap p=before.heap p

def broadcastReverse : List Nat→List Nat→Option (List Nat)
  | [],b => some b
  | a,[] => some a
  | a::as,b::bs => do
    let tail ← broadcastReverse as bs
    if a=b then some (a::tail) else if a=1 then some (b::tail)
    else if b=1 then some (a::tail) else none

def broadcastShape (a b : List Nat) := (broadcastReverse a.reverse b.reverse).map List.reverse

def logicalStrides : List Nat→List Nat
  | [] => []
  | _::rest => rest.prod::logicalStrides rest

/-- Row-major broadcast projection, accumulated from the last axis. Missing
leading input axes broadcast with extent one. -/
def broadcastIndexReverse : List Nat→List Nat→Nat→Nat
  | [],_,_ => 0
  | _,[],_ => 0
  | o::os,d::ds,i => (if d=1 then 0 else i%o)+d*broadcastIndexReverse os ds (i/o)

def broadcastIndex (out input : List Nat) (i : Nat) : Nat :=
  broadcastIndexReverse out.reverse input.reverse i

/-- Only local scalar/reduction primitives, independent of every KLDIV source
or mathematical result. The sum keeps the actual Torch device sum operation. -/
structure Numerics (Word : Type) where
  (zero one : Word)
  neg : Word→Word
  equal : Word→Word→Bool
  sum : Bool→List Word→Word
  divide : Word→Nat→Word
  multiply : Word→Word→Word

def tensorEqual {Word : Type} (codec : Codec Word) (num : Numerics Word) (m : Heap) (a b : Tensor) : Bool :=
  if a.shape=b.shape then (List.range a.numel).all
    (fun i => num.equal (value codec num.neg m a i) (value codec num.neg m b i)) else Bool.false

structure Library {Word : Type} (codec : Codec Word) (num : Numerics Word) (api : Services Word) : Prop where
  conj : ∀ a s b t,a.backing∈s.live → api.resolveConj a s=.ok b t → Materialized codec num.neg a b s t
  neg : ∀ a s b t,a.backing∈s.live → api.resolveNeg a s=.ok b t →
    Materialized codec num.neg a b s t ∧ b.negative=Bool.false
  contiguous : ∀ a s b t,a.backing∈s.live → a.negative=Bool.false → api.contiguous a s=.ok b t →
    Materialized codec num.neg a b s t ∧ Dense b
  /-- Default clone may preserve a noncontiguous dense layout. Only normal
  logical values and fresh storage are needed for the entry cotangent. -/
  clone : ∀ a s b t,a.backing∈s.live → a.negative=Bool.false → api.clone a s=.ok b t →
    Materialized codec num.neg a b s t ∧ b.negative=Bool.false ∧ (∀ r,r∈s.live → b.backing.disjoint r)
  zeros : ∀ sh device s b t,api.zeros sh device s=.ok b t →
    Allocated sh device b s t ∧ (∀ i,i<b.numel → value codec num.neg t.heap b i=num.zero)
  empty : ∀ a s b t,a.backing∈s.live → api.emptyContiguousLike a s=.ok b t →
    Allocated a.shape a.device b s t
  copy : ∀ dst src s b t,dst.backing∈s.live → src.backing∈s.live → dst.shape=src.shape →
    api.copy dst src s=.ok b t → Copied codec num.neg dst src b s t
  sum : ∀ a dim0 s b t,a.backing∈s.live → a.shape.length=1 → api.sum a dim0 s=.ok b t →
    Allocated [] a.device b s t ∧ value codec num.neg t.heap b 0=
      num.sum dim0 ((List.range a.numel).map (value codec num.neg s.heap a))
  divide : ∀ a n s b t,a.backing∈s.live → a.shape=[] → api.divNat a n s=.ok b t →
    Allocated [] a.device b s t ∧ value codec num.neg t.heap b 0=
      num.divide (value codec num.neg s.heap a 0) n
  one : ∀ device s b t,api.scalarOne device s=.ok b t →
    Allocated [] device b s t ∧ value codec num.neg t.heap b 0=num.one
  equal : ∀ a b s result t,a.backing∈s.live → b.backing∈s.live → api.equal a b s=.ok result t →
    t=s ∧ result=tensorEqual codec num s.heap a b
  multiply : ∀ a b s out t,a.backing∈s.live → b.backing∈s.live → api.multiply a b s=.ok out t →
    broadcastShape a.shape b.shape=some out.shape ∧ Allocated out.shape a.device out s t ∧
    (∀ i,i<out.numel → value codec num.neg t.heap out i=num.multiply
      (value codec num.neg s.heap a (broadcastIndex out.shape a.shape i))
      (value codec num.neg s.heap b (broadcastIndex out.shape b.shape i)))
  configure : ∀ rows cols s t,api.configure rows cols s=.ok () t → t=s

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
