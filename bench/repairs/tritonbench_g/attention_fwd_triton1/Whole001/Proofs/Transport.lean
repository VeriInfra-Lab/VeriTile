import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Source

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

def Inputs.map (f : α→β) (xs : Inputs α) : Inputs β :=
  ⟨fun h r d => f (xs.q h r d),fun h r d => f (xs.k h r d),fun h r d => f (xs.v h r d)⟩

/-- A scalar-algebra homomorphism, used only for expression evaluation.
Floating-point-to-real equality is derived separately from local laws. -/
structure Ops.Hom (a : Ops α) (b : Ops β) (f : α→β) : Prop where
  zero : f a.zero=b.zero
  cast : ∀ x, f (a.cast x)=b.cast (f x)
  mul : ∀ x y, f (a.mul x y)=b.mul (f x) (f y)
  fma : ∀ x y z, f (a.fma x y z)=b.fma (f x) (f y) (f z)

namespace Ops.Hom
variable {a : Ops α} {b : Ops β} {f : α→β} (hom : Ops.Hom a b f)
include hom

theorem chain_map (n : Nat) (left right : Nat→α) (seed : α) :
    f (chain a n left right seed)=chain b n (fun i => f (left i)) (fun i => f (right i)) (f seed) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [chain,hom.fma,ih]

theorem masked_map (c : Shape) (input : Nat→Nat→Nat→α) (h r d : Nat) :
    f (masked a c input h r d)=masked b c (fun h r d => f (input h r d)) h r d := by
  unfold masked
  split_ifs
  · rfl
  · exact hom.zero

theorem query_map (c : Shape) (xs : Inputs α) (scale : α) (h n i j : Nat) :
    f (Body.query a c xs scale h n i j)=Body.query b c (xs.map f) (f scale) h n i j := by
  simp only [Body.query,hom.cast,hom.mul,hom.masked_map,Inputs.map]
theorem key_map (c : Shape) (xs : Inputs α) (h n i j : Nat) :
    f (Body.key a c xs h n i j)=Body.key b c (xs.map f) h n i j := by
  exact hom.masked_map c xs.k h (n*32+j) i
theorem value_map (c : Shape) (xs : Inputs α) (h n i j : Nat) :
    f (Body.value a c xs h n i j)=Body.value b c (xs.map f) h n i j := by
  exact hom.masked_map c xs.v h (n*32+i) j
theorem score_map (c : Shape) (q k : Tile α) (i j : Nat) :
    f (Body.score a c q k i j)=
      Body.score b c (fun i j => f (q i j)) (fun i j => f (k i j)) i j := by
  simp only [Body.score,hom.chain_map,hom.zero]
theorem local_map (s v : Tile α) (i j : Nat) :
    f (Body.localOutput a s v i j)=
      Body.localOutput b (fun i j => f (s i j)) (fun i j => f (v i j)) i j := by
  simp only [Body.localOutput,hom.chain_map,hom.zero,hom.cast]
theorem output_map (c : Shape) (q carry within : Tile α) (first : Bool) (i j : Nat) :
    f (Body.output a c q carry within first i j)=
      Body.output b c (fun i j => f (q i j)) (fun i j => f (carry i j))
        (fun i j => f (within i j)) first i j := by
  cases first <;> simp only [Body.output,Bool.false_eq_true,↓reduceIte,hom.chain_map,hom.cast]
theorem next_map (k v carry : Tile α) (first : Bool) (i j : Nat) :
    f (Body.nextCarry a k v carry first i j)=
      Body.nextCarry b (fun i j => f (k i j)) (fun i j => f (v i j))
        (fun i j => f (carry i j)) first i j := by
  cases first <;> simp only [Body.nextCarry,Bool.false_eq_true,↓reduceIte,hom.chain_map,hom.zero]

theorem evaluate_output_map (c : Shape) (xs : Inputs α) (scale : α) (h n : Nat)
    (carry : Tile α) (ifcond : Bool) (i j : Nat) :
    f ((Body.evaluate a c xs scale h n carry ifcond).output i j)=
      (Body.evaluate b c (xs.map f) (f scale) h n (fun i j => f (carry i j)) ifcond).output i j := by
  simp only [Body.evaluate,hom.output_map,hom.query_map,hom.key_map,hom.value_map,
    hom.local_map,hom.score_map]

theorem evaluate_carry_map (c : Shape) (xs : Inputs α) (scale : α) (h n : Nat)
    (carry : Tile α) (ifcond : Bool) (i j : Nat) :
    f ((Body.evaluate a c xs scale h n carry ifcond).carry i j)=
      (Body.evaluate b c (xs.map f) (f scale) h n (fun i j => f (carry i j)) ifcond).carry i j := by
  simp only [Body.evaluate,hom.next_map,hom.key_map,hom.value_map]

theorem carry_map (c : Shape) (xs : Inputs α) (scale : α) (h : Nat) (ifcond : Bool) (n : Nat) :
    (fun i j => f (Reference.carry a c xs scale h ifcond n i j))=
      Reference.carry b c (xs.map f) (f scale) h ifcond n := by
  induction n with
  | zero => funext i j; exact hom.zero
  | succ n ih =>
    funext i j
    simp only [Reference.carry,hom.evaluate_carry_map,ih]

theorem reference_output_map (c : Shape) (xs : Inputs α) (scale : α) (h n : Nat) (ifcond : Bool)
    (i j : Nat) : f (Reference.output a c xs scale h n ifcond i j)=
      Reference.output b c (xs.map f) (f scale) h n ifcond i j := by
  simp only [Reference.output,hom.evaluate_output_map,hom.carry_map]

end Ops.Hom

def wordOps (isa : ISA) : Ops (BitVec 32) := ⟨0,id,isa.mulRN,isa.fmaRN⟩

theorem evalHom (isa : ISA) : Ops.Hom graphOps (wordOps isa) (Expr.eval isa) :=
  ⟨rfl,fun _ => rfl,fun _ _ => rfl,fun _ _ _ => rfl⟩
theorem realHom (decode : BitVec 32→ℝ) : Ops.Hom graphOps realOps (Expr.real decode) :=
  ⟨rfl,fun _ => rfl,fun _ _ => rfl,fun _ _ _ => rfl⟩

namespace Graph
def inputs (xs : Inputs (BitVec 32)) : Inputs Expr := xs.map Expr.word
def output (c : Shape) (xs : Inputs (BitVec 32)) (scale : BitVec 32)
    (head chunk : Nat) (ifcond : Bool) (i j : Nat) : Expr :=
  Reference.output graphOps c (inputs xs) (.word scale) head chunk ifcond i j
def carry (c : Shape) (xs : Inputs (BitVec 32)) (scale : BitVec 32)
    (head : Nat) (ifcond : Bool) (chunk i j : Nat) : Expr :=
  Reference.carry graphOps c (inputs xs) (.word scale) head ifcond chunk i j

theorem eval_output (isa : ISA) (c : Shape) (xs : Inputs (BitVec 32)) (scale : BitVec 32)
    (head chunk : Nat) (ifcond : Bool) (i j : Nat) :
    (output c xs scale head chunk ifcond i j).eval isa=
      Reference.output (wordOps isa) c xs scale head chunk ifcond i j := by
  exact (evalHom isa).reference_output_map c (inputs xs) (.word scale) head chunk ifcond i j
theorem real_output (decode : BitVec 32→ℝ) (c : Shape) (xs : Inputs (BitVec 32)) (scale : BitVec 32)
    (head chunk : Nat) (ifcond : Bool) (i j : Nat) :
    (output c xs scale head chunk ifcond i j).real decode=
      Reference.output realOps c (xs.map decode) (decode scale) head chunk ifcond i j := by
  exact (realHom decode).reference_output_map c (inputs xs) (.word scale) head chunk ifcond i j
theorem eval_carry (isa : ISA) (c : Shape) (xs : Inputs (BitVec 32)) (scale : BitVec 32)
    (head : Nat) (ifcond : Bool) (chunk i j : Nat) :
    (carry c xs scale head ifcond chunk i j).eval isa=
      Reference.carry (wordOps isa) c xs scale head ifcond chunk i j := by
  exact congrFun (congrFun ((evalHom isa).carry_map c (inputs xs) (.word scale) head ifcond chunk) i) j
theorem real_carry (decode : BitVec 32→ℝ) (c : Shape) (xs : Inputs (BitVec 32)) (scale : BitVec 32)
    (head : Nat) (ifcond : Bool) (chunk i j : Nat) :
    (carry c xs scale head ifcond chunk i j).real decode=
      Reference.carry realOps c (xs.map decode) (decode scale) head ifcond chunk i j := by
  exact congrFun (congrFun ((realHom decode).carry_map c (inputs xs) (.word scale) head ifcond chunk) i) j
end Graph

#axiomsClean Graph.eval_output
#axiomsClean Graph.real_output
#axiomsClean Graph.eval_carry
#axiomsClean Graph.real_carry
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
