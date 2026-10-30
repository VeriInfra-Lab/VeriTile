import bench.tritonbench_g.mixed_sparse_attention.Proofs.MixedSparseAttention_Source_Config
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000

inductive UnaryOp where
  | exp2ApproxFTZ | narrowHalf | widenHalf | narrowBF16 | widenBF16
inductive BinaryOp where
  | add32 | sub32 | mul32 | maxNum32 | divFull32 | addHalf | mulHalf
inductive TernaryOp where
  | fma32 | fmaBF16

/-- Scalar ISA operations and one coordinate of a fixed m16n8k16 instruction.
No variable-length dot, reduction, block step or final-output port exists. -/
inductive Graph where
  | bits (value : BitVec 32)
  | literal32 (value : ℝ)
  | unary (op : UnaryOp) (a : Graph)
  | binary (op : BinaryOp) (a b : Graph)
  | ternary (op : TernaryOp) (a b c : Graph)
  | mma16 (storage : TypedSource.Storage) (row : Fin 16) (col : Fin 8)
      (a b : Fin 16→ Graph) (seed : Graph)
  | shuffle (xorMask : Nat) (a : Graph)
  | shared (slot : Nat) (a : Graph)

structure ISA (α : Type) where
  bits : BitVec 32→α
  literal32 : ℝ→α
  unary : UnaryOp→α→α
  binary : BinaryOp→α→α→α
  ternary : TernaryOp→α→α→α→α
  mma16 : TypedSource.Storage→ Fin 16→ Fin 8→(Fin 16→α)→(Fin 16→α)→α→α

def Graph.eval {α : Type} (I : ISA α) : Graph→α
  | .bits b=>I.bits b
  | .literal32 x=>I.literal32 x
  | .unary op a=>I.unary op (a.eval I)
  | .binary op a b=>I.binary op (a.eval I) (b.eval I)
  | .ternary op a b c=>I.ternary op (a.eval I) (b.eval I) (c.eval I)
  | .mma16 dtype r c a b seed=>I.mma16 dtype r c (fun k=>(a k).eval I) (fun k=>(b k).eval I) (seed.eval I)
  | .shuffle _ a=>a.eval I
  | .shared _ a=>a.eval I

def zero : Graph := .bits 0

def dot16 (dtype : TypedSource.Storage) (D row col : Nat)
    (q k : Fin D→ Graph) (seed : Graph) : Graph :=
  (List.range ((D+15)/16)).foldl (fun acc block=>
    .mma16 dtype ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
      (fun d=>if h : block*16+d.val<D then q ⟨block*16+d.val,h⟩ else zero)
      (fun d=>if h : block*16+d.val<D then k ⟨block*16+d.val,h⟩ else zero) acc) seed

theorem mma_coverage (D : Nat) (d : Fin D) :
    d.val/16<(D+15)/16 ∧ d.val/16*16+d.val%16=d.val := by omega

inductive Tree where
  | leaf (index : Nat)
  | merge (a b : Tree)
  | shuffle (mask : Nat) (a : Tree)
  | shared (slot : Nat) (a : Tree)

def Tree.leaves : Tree→ List Nat
  | .leaf i=>[i]
  | .merge a b=>a.leaves++b.leaves
  | .shuffle _ a=>a.leaves
  | .shared _ a=>a.leaves

def Tree.value (combine : BinaryOp) (input : Nat→ Graph) : Tree→ Graph
  | .leaf i=>input i
  | .merge a b=>.binary combine (a.value combine input) (b.value combine input)
  | .shuffle mask a=>.shuffle mask (a.value combine input)
  | .shared slot a=>.shared slot (a.value combine input)

/-- One row is owned by four lane groups. Consecutive register pairs
cover two columns, then advance eight columns for the next MMA fragment. -/
def localIndex (owner : Fin 4) (j : Nat) : Nat := j/2*8+2*owner.val+j%2

def localTree (N : Nat) (owner : Fin 4) : Tree :=
  (List.range (N/4-1)).foldl (fun t j=>.merge t (.leaf (localIndex owner (j+1))))
    (.leaf (localIndex owner 0))

def xorOwner (owner : Fin 4) (mask : Nat) : Fin 4 :=
  ⟨(Nat.xor owner.val mask)%4,Nat.mod_lt _ (by decide)⟩

/-- Saved SM80 [4,1] MMA warp layout: the source reductions use sequential
local registers followed by XOR2 then XOR1. There is no cross-warp reduction. -/
def plan (N : Nat) (owner : Fin 4) : Tree :=
  .merge
    (.merge (localTree N owner) (.shuffle 2 (localTree N (xorOwner owner 2))))
    (.shuffle 1 (.merge (localTree N (xorOwner owner 1))
      (.shuffle 2 (localTree N (xorOwner owner 3)))))

def physicalOwner (column : Nat) : Fin 4 :=
  ⟨column%8/2,by have:=Nat.mod_lt column (by decide : 0<8);omega⟩

theorem reduction_coverage (N : Nat) (valid : N=16 ∨ N=32 ∨ N=64 ∨ N=128) (u : Fin 4) :
    (plan N u).leaves.Perm (List.range N) := by
  fin_cases u <;> rcases valid with rfl|rfl|rfl|rfl <;> unfold plan <;> decide

structure LocalTransport {α β : Type} (f : α→β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x,f (A.bits x)=B.bits x
  literal32 : ∀ x,f (A.literal32 x)=B.literal32 x
  unary : ∀ op a,f (A.unary op a)=B.unary op (f a)
  binary : ∀ op a b,f (A.binary op a b)=B.binary op (f a) (f b)
  ternary : ∀ op a b c,f (A.ternary op a b c)=B.ternary op (f a) (f b) (f c)
  mma16 : ∀ dtype r c a b seed,f (A.mma16 dtype r c a b seed)=
    B.mma16 dtype r c (fun k=>f (a k)) (fun k=>f (b k)) (f seed)

theorem transport {α β : Type} (f : α→β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (g : Graph) : f (g.eval A)=g.eval B := by
  induction g with
  | bits x=>exact h.bits x
  | literal32 x=>exact h.literal32 x
  | unary op a ha=>simp only [Graph.eval,h.unary,ha]
  | binary op a b ha hb=>simp only [Graph.eval,h.binary,ha,hb]
  | ternary op a b c ha hb hc=>simp only [Graph.eval,h.ternary,ha,hb,hc]
  | mma16 dtype r c a b seed ha hb hs=>simp only [Graph.eval,h.mma16,ha,hb,hs]
  | shuffle mask a ha=>exact ha
  | shared slot a ha=>exact ha

#axiomsClean reduction_coverage
#axiomsClean transport
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
