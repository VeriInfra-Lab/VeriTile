import bench.tritonbench_g.context_attn_bloom.Proofs.Kernels
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Native
open VeriTile.Triton
noncomputable section

/-- Ordered scalar reduction tree; no reduction result is a numerical port. -/
inductive Tree (n : Nat) where
  | input (i : Fin n)
  | node (left right : Tree n)

def Tree.leaves : Tree n → List (Fin n)
  | .input i => [i]
  | .node a b => a.leaves ++ b.leaves

def Tree.fold {α : Type} (input : Fin n → α) (op : α → α → α) : Tree n → α
  | .input i => input i
  | .node a b => op (a.fold input op) (b.fold input op)

structure OrderedTree (n : Nat) where
  tree : Tree n
  complete : tree.leaves.Perm (List.finRange n)

inductive Graph where
  | bits (word : BitVec 32)
  | literal32 (value : ℝ)
  | addRN (left right : Graph)
  | subRN (left right : Graph)
  | mulRN (left right : Graph)
  | divFull (left right : Graph)
  | maxNum (left right : Graph)
  | ex2Approx (value : Graph)
  | fmaRN (left right seed : Graph)
  | mmaTF32 (row : Fin 16) (col : Fin 8) (a b : Fin 8 → Graph) (seed : Graph)

/-- Only one scalar instruction or one output coordinate of the fixed SM80
m16n8k8 TF32 instruction may receive deferred numerical meaning. Raw input
bits and bit-preserving moves/casts are not floating arithmetic callbacks. -/
structure ISA (α : Type) where
  bits : BitVec 32 → α
  literal32 : ℝ → α
  addRN : α → α → α
  subRN : α → α → α
  mulRN : α → α → α
  divFull : α → α → α
  maxNum : α → α → α
  ex2Approx : α → α
  fmaRN : α → α → α → α
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → α) → (Fin 8 → α) → α → α

/-- Native bit words are identity-mapped; only these nine instruction ports
carry deferred numerical interpretation. -/
structure WordISA where
  literal32 : ℝ → BitVec 32
  addRN : BitVec 32 → BitVec 32 → BitVec 32
  subRN : BitVec 32 → BitVec 32 → BitVec 32
  mulRN : BitVec 32 → BitVec 32 → BitVec 32
  divFull : BitVec 32 → BitVec 32 → BitVec 32
  maxNum : BitVec 32 → BitVec 32 → BitVec 32
  ex2Approx : BitVec 32 → BitVec 32
  fmaRN : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → BitVec 32) → (Fin 8 → BitVec 32) → BitVec 32 → BitVec 32

def WordISA.interpret (I : WordISA) : ISA (BitVec 32) where
  bits := id
  literal32 := I.literal32
  addRN := I.addRN
  subRN := I.subRN
  mulRN := I.mulRN
  divFull := I.divFull
  maxNum := I.maxNum
  ex2Approx := I.ex2Approx
  fmaRN := I.fmaRN
  mmaTF32 := I.mmaTF32

def Graph.eval {α : Type} (I : ISA α) : Graph → α
  | .bits b => I.bits b
  | .literal32 x => I.literal32 x
  | .addRN a b => I.addRN (a.eval I) (b.eval I)
  | .subRN a b => I.subRN (a.eval I) (b.eval I)
  | .mulRN a b => I.mulRN (a.eval I) (b.eval I)
  | .divFull a b => I.divFull (a.eval I) (b.eval I)
  | .maxNum a b => I.maxNum (a.eval I) (b.eval I)
  | .ex2Approx a => I.ex2Approx (a.eval I)
  | .fmaRN a b c => I.fmaRN (a.eval I) (b.eval I) (c.eval I)
  | .mmaTF32 row col a b seed =>
      I.mmaTF32 row col (fun k => (a k).eval I) (fun k => (b k).eval I) (seed.eval I)

def zero : Graph := .bits 0

def dotStep (D row col block : Nat) (q k : Fin D → Graph) (seed : Graph) : Graph :=
  .mmaTF32 ⟨row % 16, Nat.mod_lt _ (by decide)⟩ ⟨col % 8, Nat.mod_lt _ (by decide)⟩
    (fun lane => if h : block*8+lane.val<D then q ⟨block*8+lane.val,h⟩ else zero)
    (fun lane => if h : block*8+lane.val<D then k ⟨block*8+lane.val,h⟩ else zero) seed

def dot (D row col : Nat) (q k : Fin D → Graph) (seed : Graph) : Graph :=
  (List.range ((D+7)/8)).foldl (fun previous block => dotStep D row col block q k previous) seed

theorem dot_feature_covered (D d : Nat) (hd : d<D) :
    d/8 ∈ List.range ((D+7)/8) ∧ d%8<8 ∧ d/8*8+d%8=d := by
  refine ⟨List.mem_range.mpr ?_,Nat.mod_lt _ (by decide),by simpa [Nat.mul_comm] using Nat.div_add_mod d 8⟩
  omega

theorem dot_feature_unique (d block lane : Nat) (hl : lane<8) (he : block*8+lane=d) :
    block=d/8 ∧ lane=d%8 := by omega

theorem reduction_covered (t : OrderedTree n) (i : Fin n) : i ∈ t.tree.leaves :=
  t.complete.mem_iff.mpr (List.mem_finRange i)

theorem reduction_nodup (t : OrderedTree n) : t.tree.leaves.Nodup :=
  t.complete.nodup_iff.mpr (List.nodup_finRange n)

/-- These are local instruction equations only. Arbitrary reduction/dot sizes
and whole outputs are transported by induction over the constructed graph. -/
structure LocalTransport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x, f (A.bits x)=B.bits x
  literal32 : ∀ x, f (A.literal32 x)=B.literal32 x
  addRN : ∀ a b, f (A.addRN a b)=B.addRN (f a) (f b)
  subRN : ∀ a b, f (A.subRN a b)=B.subRN (f a) (f b)
  mulRN : ∀ a b, f (A.mulRN a b)=B.mulRN (f a) (f b)
  divFull : ∀ a b, f (A.divFull a b)=B.divFull (f a) (f b)
  maxNum : ∀ a b, f (A.maxNum a b)=B.maxNum (f a) (f b)
  ex2Approx : ∀ a, f (A.ex2Approx a)=B.ex2Approx (f a)
  fmaRN : ∀ a b c, f (A.fmaRN a b c)=B.fmaRN (f a) (f b) (f c)
  mmaTF32 : ∀ row col a b c, f (A.mmaTF32 row col a b c)=
    B.mmaTF32 row col (fun k => f (a k)) (fun k => f (b k)) (f c)

theorem transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (g : Graph) : f (g.eval A)=g.eval B := by
  induction g with
  | bits b => exact h.bits b
  | literal32 x => exact h.literal32 x
  | addRN a b ha hb => simp only [Graph.eval,h.addRN,ha,hb]
  | subRN a b ha hb => simp only [Graph.eval,h.subRN,ha,hb]
  | mulRN a b ha hb => simp only [Graph.eval,h.mulRN,ha,hb]
  | divFull a b ha hb => simp only [Graph.eval,h.divFull,ha,hb]
  | maxNum a b ha hb => simp only [Graph.eval,h.maxNum,ha,hb]
  | ex2Approx a ha => simp only [Graph.eval,h.ex2Approx,ha]
  | fmaRN a b c ha hb hc => simp only [Graph.eval,h.fmaRN,ha,hb,hc]
  | mmaTF32 row col a b c ha hb hc => simp only [Graph.eval,h.mmaTF32,ha,hb,hc]

#axiomsClean transport
#axiomsClean dot_feature_covered
#axiomsClean reduction_covered
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Native
