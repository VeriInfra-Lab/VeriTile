import bench.tritonbench_g.masked_add_cuda.Proofs.Grid

/-! Native graph selected by Triton 3.7.1 SM80/SM90 compilation with fusion.
Multiplication is staged and either source addition order consumes it as one
FMA, with alpha first, p second, grad third. `check_lowering.py` checks both
actual variants and signed mask widths. This is a selected lowering model,
not a proof of the compiler or equality between split rounding and FMA. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.Graph
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32

inductive Expr where
  | leaf (word : Word)
  | literalValue (value : ℝ)
  | product (data alpha : Expr)
  | fma (alpha data grad : Expr)
  | other (op : Binary) (left right : Expr)
  deriving Inhabited

def binary : Binary → Expr → Expr → Expr
  | .mul, data, alpha => .product data alpha
  | .add, grad, .product data alpha => .fma alpha data grad
  | .add, .product data alpha, grad => .fma alpha data grad
  | op, left, right => .other op left right

/-- A same-format FP32 load/cast is bit-preserving. Scalar conversion is the
separate scalar leaf; integer-default-zero and Python-float ABI are explicit
native primitive profiles in the public entry. Unused operators stay opaque. -/
def algebra (rest : Algebra Expr) : Algebra Expr :=
  { rest with
    literal := fun _ _ x => Expr.literalValue x
    fp32Load := id
    cast := fun _ _ _ x => x
    binary := fun _ _ op => binary op }

/-- The native ABI already converted the Python scalar to one FP32 word.
The Real token in the generic AST names that single scalar input; it is not a
representation of Python NaN, infinity or signed zero. No other floating
literal occurs in this source body. -/
def nativeAlgebra (rest : Algebra Expr) (alphaWord : Word) : Algebra Expr :=
  { algebra rest with literal := fun _ _ _ => .leaf alphaWord }

theorem source_fma_word (rest : Algebra Expr) (alphaWord : Word) (token : ℝ) (g p : Expr) :
    updateValue (nativeAlgebra rest alphaWord) token g p = .fma (.leaf alphaWord) p g := rfl

structure Operations where
  scalar32 : ℝ → Word
  fmaRN32 : Word → Word → Word → Word
  mulRN32 : Word → Word → Word
  other : Binary → Word → Word → Word

def eval (ops : Operations) : Expr → Word
  | .leaf x => x
  | .literalValue x => ops.scalar32 x
  | .product x y => ops.mulRN32 (eval ops x) (eval ops y)
  | .fma alpha data grad => ops.fmaRN32 (eval ops alpha) (eval ops data) (eval ops grad)
  | .other op x y => ops.other op (eval ops x) (eval ops y)

theorem source_fma (rest : Algebra Expr) (alpha : ℝ) (g p : Expr) :
    updateValue (algebra rest) alpha g p = .fma (.literalValue alpha) p g := rfl

def reference (ops : Operations) (alpha : ℝ) (g p : Word) (mask : Int) : Word :=
  if mask=0 then ops.fmaRN32 (ops.scalar32 alpha) p g else g

/-- Every complete source grid returns the one fused native expression at
active elements and preserves vetoed words. There is no numeric premise. -/
theorem grid_words (rest : Algebra Expr) (ops : Operations)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (s : State Expr)
    (positive : 0<p.block) (grad data : Nat → Word)
    (hg : ∀ i, i<p.n → (s.mem p.output i).read .real = .leaf (grad i))
    (hp : ∀ i, i<p.n → (s.mem p.x i).read .real = .leaf (data i)) :
    ∃ final, runGrid (algebra rest) storage p alpha s=some final ∧
      (∀ i, i<p.n → eval ops ((final.mem p.output i).read .real) =
        reference ops alpha (grad i) (data i) ((s.mem p.y i).read .int)) ∧
      (∀ r i, (r ≠ p.output ∨ p.n ≤ i ∨ (s.mem p.y i).read .int ≠ 0) →
        final.mem r i=s.mem r i) := by
  obtain ⟨final,he,hm⟩ := grid_correct (algebra rest) storage p alpha s positive
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    by_cases zero : (s.mem p.y i).read .int=0
    · simp [gridExpectedMemory,hi,zero,resultCell,algebra,updateValue,alphaValue,
        binary,eval,hg i hi,hp i hi,reference]
    · simp [gridExpectedMemory,hi,zero,eval,hg i hi,reference]
  · intro r i outside
    rw [hm]
    rcases outside with hr|hi|hz
    · simp [gridExpectedMemory,hr]
    · simp [gridExpectedMemory,Nat.not_lt.mpr hi]
    · simp [gridExpectedMemory,hz]

#specNonCircular reference avoiding [candidateKernel,FP.Structural.exec,runGrid]
#axiomsClean source_fma
#axiomsClean source_fma_word
#axiomsClean grid_words
end
end VeriTile.Bench.Repairs.MaskedAddCuda.Graph
