import bench.tritonbench_g.attn_fwd_causal.Proofs.Engine
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Native
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000

/-- Numerical ports are scalar PTX operations and one output coordinate of a
fixed SM80 instruction. There is no variable-K dot or reduction port. -/
inductive Graph where
  | bits (value : BitVec 32)
  | literal32 (value : ℝ)
  | add32 (a b : Graph)
  | sub32 (a b : Graph)
  | mul32 (a b : Graph)
  | maxNum32 (a b : Graph)
  | divFull32 (a b : Graph)
  | exp2ApproxFTZ (a : Graph)
  | narrow16 (a : Graph)
  | widen16 (a : Graph)
  | fma32 (a b c : Graph)
  | mmaTF32 (row : Fin 16) (col : Fin 8) (a b : Fin 8 → Graph) (seed : Graph)
  | mmaHalf (row : Fin 16) (col : Fin 8) (a b : Fin 16 → Graph) (seed : Graph)
  | shuffle (xorMask : Nat) (a : Graph)

structure ISA (α : Type) where
  bits : BitVec 32 → α
  literal32 : ℝ → α
  add32 : α → α → α
  sub32 : α → α → α
  mul32 : α → α → α
  maxNum32 : α → α → α
  divFull32 : α → α → α
  exp2ApproxFTZ : α → α
  narrow16 : α → α
  widen16 : α → α
  fma32 : α → α → α → α
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → α) → (Fin 8 → α) → α → α
  mmaHalf : Fin 16 → Fin 8 → (Fin 16 → α) → (Fin 16 → α) → α → α

def Graph.eval {α : Type} (I : ISA α) : Graph → α
  | .bits b=>I.bits b
  | .literal32 x=>I.literal32 x
  | .add32 a b=>I.add32 (a.eval I) (b.eval I)
  | .sub32 a b=>I.sub32 (a.eval I) (b.eval I)
  | .mul32 a b=>I.mul32 (a.eval I) (b.eval I)
  | .maxNum32 a b=>I.maxNum32 (a.eval I) (b.eval I)
  | .divFull32 a b=>I.divFull32 (a.eval I) (b.eval I)
  | .exp2ApproxFTZ a=>I.exp2ApproxFTZ (a.eval I)
  | .narrow16 a=>I.narrow16 (a.eval I)
  | .widen16 a=>I.widen16 (a.eval I)
  | .fma32 a b c=>I.fma32 (a.eval I) (b.eval I) (c.eval I)
  | .mmaTF32 r c a b seed=>I.mmaTF32 r c (fun k=>(a k).eval I) (fun k=>(b k).eval I) (seed.eval I)
  | .mmaHalf r c a b seed=>I.mmaHalf r c (fun k=>(a k).eval I) (fun k=>(b k).eval I) (seed.eval I)
  | .shuffle _ a=>a.eval I

def zero : Graph := .bits 0

def dotTF32 (D row col : Nat) (q k : Fin D → Graph) : Graph :=
  (List.range ((D+7)/8)).foldl (fun seed block=>
    .mmaTF32 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
      (fun d=>if h : block*8+d.val<D then q ⟨block*8+d.val,h⟩ else zero)
      (fun d=>if h : block*8+d.val<D then k ⟨block*8+d.val,h⟩ else zero) seed) zero

def dotHalf (D row col : Nat) (q k : Fin D → Graph) : Graph :=
  (List.range ((D+15)/16)).foldl (fun seed block=>
    .mmaHalf ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
      (fun d=>if h : block*16+d.val<D then q ⟨block*16+d.val,h⟩ else zero)
      (fun d=>if h : block*16+d.val<D then k ⟨block*16+d.val,h⟩ else zero) seed) zero

theorem tf32_coverage (d : Fin 128) : d.val/8<16 ∧ d.val/8*8+d.val%8=d.val := by omega
theorem half_coverage (d : Fin 64) : d.val/16<4 ∧ d.val/16*16+d.val%16=d.val := by omega

/-- Actual row reduction: each of four lanes folds its 16 local values, then
exchanges with xor 2 and xor 1. Replica trees remain distinct. -/
inductive Tree where
  | leaf (index : Fin 64)
  | merge (a b : Tree)
  | shuffle (mask : Nat) (a : Tree)

def Tree.leaves : Tree → List Nat
  | .leaf i=>[i.val]
  | .merge a b=>a.leaves++b.leaves
  | .shuffle _ a=>a.leaves

def Tree.value (merge : Graph → Graph → Graph) (input : Fin 64 → Graph) : Tree → Graph
  | .leaf i=>input i
  | .merge a b=>merge (a.value merge input) (b.value merge input)
  | .shuffle mask a=>.shuffle mask (a.value merge input)

def localIndex (owner : Fin 4) (j : Fin 16) : Fin 64 :=
  ⟨j.val/2*8+2*owner.val+j.val%2,by have:=owner.isLt;have:=j.isLt;omega⟩
def localTree (owner : Fin 4) : Tree :=
  (List.finRange 15).foldl (fun t j=>.merge t (.leaf (localIndex owner ⟨j.val+1,by omega⟩)))
    (.leaf (localIndex owner ⟨0,by decide⟩))
def xorOwner (owner : Fin 4) (mask : Nat) : Fin 4 :=
  ⟨(owner.val ^^^ mask)%4,Nat.mod_lt _ (by decide)⟩
def plan (owner : Fin 4) : Tree :=
  .merge
    (.merge (localTree owner) (.shuffle 2 (localTree (xorOwner owner 2))))
    (.shuffle 1 (.merge (localTree (xorOwner owner 1))
      (.shuffle 2 (localTree (xorOwner owner 3)))))

def physicalOwner (column : Nat) : Fin 4 := ⟨column%8/2,by have:=Nat.mod_lt column (by decide : 0<8);omega⟩

theorem reduction_coverage (owner : Fin 4) : (plan owner).leaves.Perm (List.range 64) := by
  fin_cases owner <;> decide

structure LocalTransport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x,f (A.bits x)=B.bits x
  literal32 : ∀ x,f (A.literal32 x)=B.literal32 x
  add32 : ∀ a b,f (A.add32 a b)=B.add32 (f a) (f b)
  sub32 : ∀ a b,f (A.sub32 a b)=B.sub32 (f a) (f b)
  mul32 : ∀ a b,f (A.mul32 a b)=B.mul32 (f a) (f b)
  maxNum32 : ∀ a b,f (A.maxNum32 a b)=B.maxNum32 (f a) (f b)
  divFull32 : ∀ a b,f (A.divFull32 a b)=B.divFull32 (f a) (f b)
  exp2ApproxFTZ : ∀ a,f (A.exp2ApproxFTZ a)=B.exp2ApproxFTZ (f a)
  narrow16 : ∀ a,f (A.narrow16 a)=B.narrow16 (f a)
  widen16 : ∀ a,f (A.widen16 a)=B.widen16 (f a)
  fma32 : ∀ a b c,f (A.fma32 a b c)=B.fma32 (f a) (f b) (f c)
  mmaTF32 : ∀ r c a b seed,f (A.mmaTF32 r c a b seed)=B.mmaTF32 r c (fun k=>f (a k)) (fun k=>f (b k)) (f seed)
  mmaHalf : ∀ r c a b seed,f (A.mmaHalf r c a b seed)=B.mmaHalf r c (fun k=>f (a k)) (fun k=>f (b k)) (f seed)

theorem transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (g : Graph) : f (g.eval A)=g.eval B := by
  induction g with
  | bits x=>exact h.bits x
  | literal32 x=>exact h.literal32 x
  | add32 a b ha hb=>simp only [Graph.eval,h.add32,ha,hb]
  | sub32 a b ha hb=>simp only [Graph.eval,h.sub32,ha,hb]
  | mul32 a b ha hb=>simp only [Graph.eval,h.mul32,ha,hb]
  | maxNum32 a b ha hb=>simp only [Graph.eval,h.maxNum32,ha,hb]
  | divFull32 a b ha hb=>simp only [Graph.eval,h.divFull32,ha,hb]
  | exp2ApproxFTZ a ha=>simp only [Graph.eval,h.exp2ApproxFTZ,ha]
  | narrow16 a ha=>simp only [Graph.eval,h.narrow16,ha]
  | widen16 a ha=>simp only [Graph.eval,h.widen16,ha]
  | fma32 a b c ha hb hc=>simp only [Graph.eval,h.fma32,ha,hb,hc]
  | mmaTF32 r c a b seed ha hb hs=>simp only [Graph.eval,h.mmaTF32,ha,hb,hs]
  | mmaHalf r c a b seed ha hb hs=>simp only [Graph.eval,h.mmaHalf,ha,hb,hs]
  | shuffle mask a ha=>exact ha

#axiomsClean reduction_coverage
#axiomsClean transport
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Native
