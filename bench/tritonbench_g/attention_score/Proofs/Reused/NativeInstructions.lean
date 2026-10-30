import bench.tritonbench_g.attention_score.Proofs.Reused.NativeReduction

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeInstructions
open VeriTile.Triton
noncomputable section

/-- Only FP32 scalar instructions and one output coordinate of the fixed
sm80 m16n8k8 TF32 instruction are numerical ports. A dot or reduction of
arbitrary size is constructed from these nodes and is never a numerical port. -/
inductive Graph where
  | bits (word : BitVec 32)
  | literal32 (value : ℝ)
  | addRN (left right : Graph)
  | mulRN (left right : Graph)
  | negateSign (value : Graph)
  | ex2ApproxFTZ (value : Graph)
  | fmaRN (left right seed : Graph)
  | mmaTF32 (row : Fin 16) (col : Fin 8) (a b : Fin 8 → Graph) (seed : Graph)
  | move (route : NativeReduction.Route) (value : Graph)

structure ISA (α : Type) where
  bits : BitVec 32 → α
  literal32 : ℝ → α
  addRN : α → α → α
  mulRN : α → α → α
  negateSign : α → α
  ex2ApproxFTZ : α → α
  fmaRN : α → α → α → α
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → α) → (Fin 8 → α) → α → α

def Graph.eval {α : Type} (I : ISA α) : Graph → α
  | .bits b => I.bits b
  | .literal32 x => I.literal32 x
  | .addRN a b => I.addRN (a.eval I) (b.eval I)
  | .mulRN a b => I.mulRN (a.eval I) (b.eval I)
  | .negateSign a => I.negateSign (a.eval I)
  | .ex2ApproxFTZ a => I.ex2ApproxFTZ (a.eval I)
  | .fmaRN a b c => I.fmaRN (a.eval I) (b.eval I) (c.eval I)
  | .mmaTF32 row col a b seed =>
      I.mmaTF32 row col (fun k => (a k).eval I) (fun k => (b k).eval I) (seed.eval I)
  | .move _ a => a.eval I

def zero : Graph := .bits 0

def dotStep (D row col block : Nat) (q k : Fin D → Graph) (seed : Graph) : Graph :=
  .mmaTF32 ⟨row % 16, Nat.mod_lt _ (by decide)⟩ ⟨col % 8, Nat.mod_lt _ (by decide)⟩
    (fun lane => if h : block*8+lane.val<D then q ⟨block*8+lane.val,h⟩ else zero)
    (fun lane => if h : block*8+lane.val<D then k ⟨block*8+lane.val,h⟩ else zero) seed

def dot (D row col : Nat) (q k : Fin D → Graph) (seed : Graph) : Graph :=
  (List.range ((D+7)/8)).foldl (fun previous block => dotStep D row col block q k previous) seed

theorem feature_owner (D d : Nat) (hd : d<D) :
    d/8 ∈ List.range ((D+7)/8) ∧ d%8<8 ∧ d/8*8+d%8=d := by
  refine ⟨List.mem_range.mpr ?_, Nat.mod_lt _ (by decide), by simpa [Nat.mul_comm] using Nat.div_add_mod d 8⟩
  omega

theorem feature_unique (d block lane : Nat) (hl : lane<8) (he : block*8+lane=d) :
    block=d/8 ∧ lane=d%8 := by omega

def reduce (profile : Fin 3) (values : Nat → Graph) : Graph :=
  (NativeReduction.plan profile).value zero values Graph.move Graph.addRN

/-- These equations mention only one scalar instruction or one fixed MMA
output coordinate. Whole-program transport follows by graph induction. -/
structure LocalTransport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x, f (A.bits x)=B.bits x
  literal32 : ∀ x, f (A.literal32 x)=B.literal32 x
  addRN : ∀ a b, f (A.addRN a b)=B.addRN (f a) (f b)
  mulRN : ∀ a b, f (A.mulRN a b)=B.mulRN (f a) (f b)
  negateSign : ∀ a, f (A.negateSign a)=B.negateSign (f a)
  ex2ApproxFTZ : ∀ a, f (A.ex2ApproxFTZ a)=B.ex2ApproxFTZ (f a)
  fmaRN : ∀ a b c, f (A.fmaRN a b c)=B.fmaRN (f a) (f b) (f c)
  mmaTF32 : ∀ row col a b c, f (A.mmaTF32 row col a b c)=
    B.mmaTF32 row col (fun k => f (a k)) (fun k => f (b k)) (f c)

theorem transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (g : Graph) : f (g.eval A)=g.eval B := by
  induction g with
  | bits b => exact h.bits b
  | literal32 x => exact h.literal32 x
  | addRN a b ha hb => simp only [Graph.eval,h.addRN,ha,hb]
  | mulRN a b ha hb => simp only [Graph.eval,h.mulRN,ha,hb]
  | negateSign a ha => simp only [Graph.eval,h.negateSign,ha]
  | ex2ApproxFTZ a ha => simp only [Graph.eval,h.ex2ApproxFTZ,ha]
  | fmaRN a b c ha hb hc => simp only [Graph.eval,h.fmaRN,ha,hb,hc]
  | mmaTF32 row col a b c ha hb hc => simp only [Graph.eval,h.mmaTF32,ha,hb,hc]
  | move route a ha => exact ha

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.NativeInstructions
