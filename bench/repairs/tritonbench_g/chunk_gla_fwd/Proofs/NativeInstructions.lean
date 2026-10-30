import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeReduction

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open VeriTile Triton FP.Structural
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
noncomputable section

/-- Native graph nodes have only scalar operations and one result lane of the
fixed sm80 m16n8k8.f32.tf32.tf32.f32 instruction. There is no variable-K dot or
whole-reduction numerical port. The instruction lane consumes precisely its
eight A-row and eight B-column operands and its own FP32 accumulator. -/
inductive Graph where
  | bits (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (a b : Graph)
  | unary (precision : Option ComputeDType) (op : Unary) (a : Graph)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (a : Graph)
  | fma (a b acc : Graph)
  | mmaTF32 (laneM : Fin 16) (laneN : Fin 8) (a b : Fin 8 → Graph) (acc : Graph)
  | move (route : NativeReduction.Route) (a : Graph)

structure ISA (α : Type) where
  bits : BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  unary : Option ComputeDType → Unary → α → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  fma : α → α → α → α
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → α) → (Fin 8 → α) → α → α

def Graph.eval {α : Type} (I : ISA α) : Graph → α
  | .bits b => I.bits b
  | .literal p d x => I.literal p d x
  | .binary p d op a b => I.binary p d op (a.eval I) (b.eval I)
  | .unary p op a => I.unary p op (a.eval I)
  | .cast p s t a => I.cast p s t (a.eval I)
  | .fma a b c => I.fma (a.eval I) (b.eval I) (c.eval I)
  | .mmaTF32 lm ln a b c => I.mmaTF32 lm ln (fun j => (a j).eval I) (fun j => (b j).eval I) (c.eval I)
  | .move _ a => a.eval I

def zero : Graph := .literal (some .fp32) .real 0

def tf32Step (k row col block : Nat) (a b : Fin k → Graph) (acc : Graph) : Graph :=
  .mmaTF32 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨col%8,Nat.mod_lt _ (by decide)⟩
    (fun e => if h : block*8+e.val<k then a ⟨block*8+e.val,h⟩ else zero)
    (fun e => if h : block*8+e.val<k then b ⟨block*8+e.val,h⟩ else zero) acc

def tf32Dot (k row col : Nat) (a b : Fin k → Graph) (seed : Graph) : Graph :=
  (List.range ((k+7)/8)).foldl (fun acc block => tf32Step k row col block a b acc) seed

def ieeeDot (k : Nat) (a b : Fin k → Graph) (seed : Graph) : Graph :=
  (List.finRange k).foldl (fun acc e => .fma (a e) (b e) acc) seed

theorem tf32_feature_owner (k e : Nat) (he : e<k) :
    e/8 ∈ List.range ((k+7)/8) ∧ e%8<8 ∧ e/8*8+e%8=e := by
  refine ⟨List.mem_range.mpr ?_,Nat.mod_lt _ (by decide),by simpa [Nat.mul_comm] using Nat.div_add_mod e 8⟩
  omega

theorem tf32_feature_unique (e block lane : Nat) (hl : lane<8)
    (h : block*8+lane=e) : block=e/8 ∧ lane=e%8 := by omega

theorem ieee_features (k : Nat) : (List.finRange k).map Fin.val=List.range k := by
  simp

/-- Replacing only numerical interpretations transports the complete graph.
The hypotheses are all scalar or one fixed ISA instruction lane. -/
structure LocalTransport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β) : Prop where
  bits : ∀ x, f (A.bits x)=B.bits x
  literal : ∀ p d x, f (A.literal p d x)=B.literal p d x
  binary : ∀ p d op a b, f (A.binary p d op a b)=B.binary p d op (f a) (f b)
  unary : ∀ p op a, f (A.unary p op a)=B.unary p op (f a)
  cast : ∀ p s t a, f (A.cast p s t a)=B.cast p s t (f a)
  fma : ∀ a b c, f (A.fma a b c)=B.fma (f a) (f b) (f c)
  mmaTF32 : ∀ lm ln a b c, f (A.mmaTF32 lm ln a b c)=
    B.mmaTF32 lm ln (fun j => f (a j)) (fun j => f (b j)) (f c)

theorem transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (g : Graph) : f (g.eval A)=g.eval B := by
  induction g with
  | bits x => exact h.bits x
  | literal p d x => exact h.literal p d x
  | binary p d op a b ia ib => simp only [Graph.eval,h.binary,ia,ib]
  | unary p op a ia => simp only [Graph.eval,h.unary,ia]
  | cast p s t a ia => simp only [Graph.eval,h.cast,ia]
  | fma a b c ia ib ic => simp only [Graph.eval,h.fma,ia,ib,ic]
  | mmaTF32 lm ln a b c ia ib ic => simp only [Graph.eval,h.mmaTF32,ia,ib,ic]
  | move r a ia => exact ia

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
