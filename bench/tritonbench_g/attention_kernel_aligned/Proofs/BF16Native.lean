import bench.tritonbench_g.attention_kernel_aligned.Proofs.NativePlan
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Native
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
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
  | narrowBF16 (a : Graph)
  | widenBF16 (a : Graph)
  | fma32 (a b c : Graph)
  | fmaBF16 (a b c : Graph)
  | mmaBF32 (row : Fin 16) (col : Fin 8) (a b : Fin 16 → Graph) (seed : Graph)
  | shuffle (xorMask : Nat) (a : Graph)
  | shared (slot : Nat) (a : Graph)

structure ISA (α : Type) where
  bits : BitVec 32 → α
  literal32 : ℝ → α
  add32 : α → α → α
  sub32 : α → α → α
  mul32 : α → α → α
  maxNum32 : α → α → α
  divFull32 : α → α → α
  exp2ApproxFTZ : α → α
  narrowBF16 : α → α
  widenBF16 : α → α
  fma32 : α → α → α → α
  fmaBF16 : α → α → α → α
  mmaBF32 : Fin 16 → Fin 8 → (Fin 16 → α) → (Fin 16 → α) → α → α

def Graph.eval {α : Type} (I : ISA α) : Graph → α
  | .bits b=>I.bits b
  | .literal32 x=>I.literal32 x
  | .add32 a b=>I.add32 (a.eval I) (b.eval I)
  | .sub32 a b=>I.sub32 (a.eval I) (b.eval I)
  | .mul32 a b=>I.mul32 (a.eval I) (b.eval I)
  | .maxNum32 a b=>I.maxNum32 (a.eval I) (b.eval I)
  | .divFull32 a b=>I.divFull32 (a.eval I) (b.eval I)
  | .exp2ApproxFTZ a=>I.exp2ApproxFTZ (a.eval I)
  | .narrowBF16 a=>I.narrowBF16 (a.eval I)
  | .widenBF16 a=>I.widenBF16 (a.eval I)
  | .fma32 a b c=>I.fma32 (a.eval I) (b.eval I) (c.eval I)
  | .fmaBF16 a b c=>I.fmaBF16 (a.eval I) (b.eval I) (c.eval I)
  | .mmaBF32 r c a b seed=>I.mmaBF32 r c (fun k=>(a k).eval I) (fun k=>(b k).eval I) (seed.eval I)
  | .shuffle _ a=>a.eval I
  | .shared _ a=>a.eval I

def zero : Graph := .bits 0

def dotBF32 (D row col : Nat) (q k : Fin D → Graph) (initial : Graph) : Graph :=
  (List.range ((D+15)/16)).foldl (fun seed block=>
    .mmaBF32 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
      (fun d=>if h : block*16+d.val<D then q ⟨block*16+d.val,h⟩ else zero)
      (fun d=>if h : block*16+d.val<D then k ⟨block*16+d.val,h⟩ else zero) seed) initial

theorem half_coverage (d : Fin 64) : d.val/16<4 ∧ d.val/16*16+d.val%16=d.val := by omega

/-- Actual row reduction: each of four lanes folds its 16 local values, then
exchanges with xor 2 and xor 1. Replica trees remain distinct. -/
abbrev Tree := VeriTile.Bench.Repairs.AttentionKernelAligned.Native.Tree
abbrev selectedPlan := VeriTile.Bench.Repairs.AttentionKernelAligned.Native.selectedPlan
def physicalOwner (column : Nat) : Fin 5 :=
  ⟨column%8/2,by have:=Nat.mod_lt column (by decide : 0<8);omega⟩

def treeValue (merge : Graph → Graph → Graph) (input : Fin 64 → Graph) : Tree → Graph
  | .leaf i=>input i
  | .merge a b=>merge (treeValue merge input a) (treeValue merge input b)
  | .shuffle mask a=>.shuffle mask (treeValue merge input a)
  | .shared slot a=>.shared slot (treeValue merge input a)

/-- Evaluate the selected finite integer plan using this precision's graph. -/
def familyValue (merge : Graph → Graph → Graph) (input : Nat → Graph) :
    VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.Tree → Graph
  | .leaf i=>input i
  | .merge a b=>merge (familyValue merge input a) (familyValue merge input b)
  | .shuffle mask a=>.shuffle mask (familyValue merge input a)
  | .shared slot a=>.shared slot (familyValue merge input a)

structure LocalTransport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x,f (A.bits x)=B.bits x
  literal32 : ∀ x,f (A.literal32 x)=B.literal32 x
  add32 : ∀ a b,f (A.add32 a b)=B.add32 (f a) (f b)
  sub32 : ∀ a b,f (A.sub32 a b)=B.sub32 (f a) (f b)
  mul32 : ∀ a b,f (A.mul32 a b)=B.mul32 (f a) (f b)
  maxNum32 : ∀ a b,f (A.maxNum32 a b)=B.maxNum32 (f a) (f b)
  divFull32 : ∀ a b,f (A.divFull32 a b)=B.divFull32 (f a) (f b)
  exp2ApproxFTZ : ∀ a,f (A.exp2ApproxFTZ a)=B.exp2ApproxFTZ (f a)
  narrowBF16 : ∀ a,f (A.narrowBF16 a)=B.narrowBF16 (f a)
  widenBF16 : ∀ a,f (A.widenBF16 a)=B.widenBF16 (f a)
  fma32 : ∀ a b c,f (A.fma32 a b c)=B.fma32 (f a) (f b) (f c)
  fmaBF16 : ∀ a b c,f (A.fmaBF16 a b c)=B.fmaBF16 (f a) (f b) (f c)
  mmaBF32 : ∀ r c a b seed,f (A.mmaBF32 r c a b seed)=B.mmaBF32 r c (fun k=>f (a k)) (fun k=>f (b k)) (f seed)

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
  | narrowBF16 a ha=>simp only [Graph.eval,h.narrowBF16,ha]
  | widenBF16 a ha=>simp only [Graph.eval,h.widenBF16,ha]
  | fma32 a b c ha hb hc=>simp only [Graph.eval,h.fma32,ha,hb,hc]
  | fmaBF16 a b c ha hb hc=>simp only [Graph.eval,h.fmaBF16,ha,hb,hc]
  | mmaBF32 r c a b seed ha hb hs=>simp only [Graph.eval,h.mmaBF32,ha,hb,hs]
  | shuffle mask a ha=>exact ha
  | shared slot a ha=>exact ha

#axiomsClean transport
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Native
