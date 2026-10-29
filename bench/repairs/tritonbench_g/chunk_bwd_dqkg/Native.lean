import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedEngine
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
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
  | exp2Approx (flushSubnormal : Bool) (a : Graph)
  | neg32 (a : Graph)
  | fma32 (a b c : Graph)
  | mmaTF32 (row : Fin 16) (col : Fin 8) (a b : Fin 8 → Graph) (seed : Graph)
  | shuffle (xorMask : Nat) (a : Graph)

structure ISA (α : Type) where
  bits : BitVec 32 → α
  literal32 : ℝ → α
  add32 : α → α → α
  sub32 : α → α → α
  mul32 : α → α → α
  exp2Approx : Bool → α → α
  neg32 : α → α
  fma32 : α → α → α → α
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → α) → (Fin 8 → α) → α → α

def Graph.eval {α : Type} (I : ISA α) : Graph → α
  | .bits b=>I.bits b
  | .literal32 x=>I.literal32 x
  | .add32 a b=>I.add32 (a.eval I) (b.eval I)
  | .sub32 a b=>I.sub32 (a.eval I) (b.eval I)
  | .mul32 a b=>I.mul32 (a.eval I) (b.eval I)
  | .exp2Approx ftz a=>I.exp2Approx ftz (a.eval I)
  | .neg32 a=>I.neg32 (a.eval I)
  | .fma32 a b c=>I.fma32 (a.eval I) (b.eval I) (c.eval I)
  | .mmaTF32 r c a b seed=>I.mmaTF32 r c (fun k=>(a k).eval I) (fun k=>(b k).eval I) (seed.eval I)
  | .shuffle _ a=>a.eval I

def zero : Graph := .bits 0

def dotTF32 (D row col : Nat) (q k : Fin D → Graph) (initial : Graph) : Graph :=
  (List.range ((D+7)/8)).foldl (fun seed block=>
    .mmaTF32 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
      (fun d=>if h : block*8+d.val<D then q ⟨block*8+d.val,h⟩ else zero)
      (fun d=>if h : block*8+d.val<D then k ⟨block*8+d.val,h⟩ else zero) seed) initial

theorem tf32_coverage (D : Nat) (d : Fin D) :
    d.val/8<(D+7)/8 ∧ d.val/8*8+d.val%8=d.val := by have:=d.isLt;omega

structure LocalTransport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x,f (A.bits x)=B.bits x
  literal32 : ∀ x,f (A.literal32 x)=B.literal32 x
  add32 : ∀ a b,f (A.add32 a b)=B.add32 (f a) (f b)
  sub32 : ∀ a b,f (A.sub32 a b)=B.sub32 (f a) (f b)
  mul32 : ∀ a b,f (A.mul32 a b)=B.mul32 (f a) (f b)
  exp2Approx : ∀ ftz a,f (A.exp2Approx ftz a)=B.exp2Approx ftz (f a)
  neg32 : ∀ a,f (A.neg32 a)=B.neg32 (f a)
  fma32 : ∀ a b c,f (A.fma32 a b c)=B.fma32 (f a) (f b) (f c)
  mmaTF32 : ∀ r c a b seed,f (A.mmaTF32 r c a b seed)=B.mmaTF32 r c (fun k=>f (a k)) (fun k=>f (b k)) (f seed)

theorem transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (g : Graph) : f (g.eval A)=g.eval B := by
  induction g with
  | bits x=>exact h.bits x
  | literal32 x=>exact h.literal32 x
  | add32 a b ha hb=>simp only [Graph.eval,h.add32,ha,hb]
  | sub32 a b ha hb=>simp only [Graph.eval,h.sub32,ha,hb]
  | mul32 a b ha hb=>simp only [Graph.eval,h.mul32,ha,hb]
  | exp2Approx ftz a ha=>simp only [Graph.eval,h.exp2Approx,ha]
  | neg32 a ha=>simp only [Graph.eval,h.neg32,ha]
  | fma32 a b c ha hb hc=>simp only [Graph.eval,h.fma32,ha,hb,hc]
  | mmaTF32 r c a b seed ha hb hs=>simp only [Graph.eval,h.mmaTF32,ha,hb,hs]
  | shuffle mask a ha=>exact ha

#axiomsClean tf32_coverage
#axiomsClean transport
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Native
