import bench.tritonbench_g.batched_vecmat_mult.Proofs.NativeContract

/-! A local numerical reduction port for one output and one K block. It takes
only two vectors of raw FP32 words, one live carry, and an explicit valid tree.
It cannot inspect memory, launch state, another output, or the final result.
Every shared-writer candidate remains a distinct allowed numerical choice.
The theorem below factors the existing complete source/host proof through
these local ports; no compiler/runtime implementation proof is added. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph.Local
open VeriTile.Triton FP.Structural FP.Equational
open ValueProofs.Host (Allocator Arguments prepare TensorABI ReturnedTensor)
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Drop address tags and represent every leaf by its actual word. This does
not round a product, and it preserves every arithmetic constructor. -/
def wordLeaves : Expr → Expr
  | .word v | .input _ _ v => .word v
  | .zero => .word (FP.Value.zero .fp32)
  | .mul a b => .mul (wordLeaves a) (wordLeaves b)
  | .add a b => .add (wordLeaves a) (wordLeaves b)
  | .fma a b c => .fma (wordLeaves a) (wordLeaves b) (wordLeaves c)

theorem wordLeaves_contractAdd (a b : Expr) :
    wordLeaves (contractAdd a b) = contractAdd (wordLeaves a) (wordLeaves b) := by
  cases a <;> cases b <;> rfl

theorem wordLeaves_contract (e : Expr) : wordLeaves (contract e) = contract (wordLeaves e) := by
  induction e with
  | word v => rfl
  | input r o v => rfl
  | zero => rfl
  | mul a b ha hb => simp [contract, wordLeaves, ha, hb]
  | add a b ha hb => simp [contract, wordLeaves, wordLeaves_contractAdd, ha, hb]
  | fma a b c ha hb hc => simp [contract, wordLeaves, ha, hb, hc]

theorem evaluate_wordLeaves (cfg : FP.Config) (e : Expr) :
    evaluate cfg (wordLeaves e) = evaluate cfg e := by
  induction e <;> simp_all [wordLeaves, evaluate]

def productTree (a b : Fin n → ValueProofs.Word) : ReductionTree n → Expr
  | .input i => .mul (.word (a i)) (.word (b i))
  | .zero => .word (FP.Value.zero .fp32)
  | .add x y => .add (productTree a b x) (productTree a b y)

/-- One local contracted reduction update. The incoming carry remains live;
BK1 is FMA on raw A/B/carry, and larger trees retain the selected contractions.
The domain is all FP32 words, including infinities, NaNs and subnormals. -/
def dotAcc (cfg : FP.Config) (plan : ReductionPlan n)
    (a b : Fin n → ValueProofs.Word) (carry : ValueProofs.Word) : ValueProofs.Word :=
  evaluate cfg (contract (.add (.word carry) (productTree a b plan.tree)))

/-- A local primitive interface, analogous to a dot-with-accumulator port.
The tree argument records a numerical choice; it does not assert equality of
different choices. -/
abbrev DotAcc := ∀ n, ReductionPlan n → (Fin n → ValueProofs.Word) →
  (Fin n → ValueProofs.Word) → ValueProofs.Word → ValueProofs.Word

def PrimitiveLaw (cfg : FP.Config) (D : DotAcc) : Prop :=
  ∀ n (plan : ReductionPlan n) a b carry, D n plan a b carry = dotAcc cfg plan a b carry

/-- The finite candidate result relation keeps every recorded writer tree.
No equality between two witnesses is required or derivable in general. -/
def PossibleResult (cfg : FP.Config) (allowed : List (ReductionPlan n))
    (a b : Fin n → ValueProofs.Word) (carry result : ValueProofs.Word) : Prop :=
  ∃ plan ∈ allowed, result = dotAcc cfg plan a b carry

/-- Native implementations may expose several numerical results for the same
local operands. This relation has no memory, program or ownership argument. -/
abbrev ReductionResults := ∀ n, (Fin n → ValueProofs.Word) →
  (Fin n → ValueProofs.Word) → ValueProofs.Word → ValueProofs.Word → Prop

/-- The exact local numerical conformance obligation for such an implementation.
It permits every listed alternative, without requiring deterministic selection
or equality between the alternatives. No implementation proof is asserted. -/
def ResultLaw (cfg : FP.Config) (allowed : ∀ n, List (ReductionPlan n))
    (nativeResults : ReductionResults) : Prop :=
  ∀ n a b carry result, nativeResults n a b carry result →
    PossibleResult cfg (allowed n) a b carry result

def capturedChoices (n : Nat) : List (ReductionPlan n) :=
  if h : n = 8 then h.symm ▸ sm90Shared8Choices else
  if h : n = 16 then h.symm ▸ sm90Shared16Choices else [sm90Plans n]

theorem every_captured_choice_possible (cfg : FP.Config) (n : Nat)
    (plan : ReductionPlan n) (h : plan ∈ capturedChoices n) a b carry :
    PossibleResult cfg (capturedChoices n) a b carry (dotAcc cfg plan a b carry) :=
  ⟨plan,h,rfl⟩

theorem captured_choice_counts : (capturedChoices 8).length = 8 ∧
    (capturedChoices 16).length = 16 := by decide +kernel

private def carryForm : Expr → Prop
  | .mul _ _ => False
  | _ => True

private theorem contractAdd_carryForm (a b : Expr) : carryForm (contractAdd a b) := by
  cases a <;> cases b <;> trivial

private theorem tensor_carryForm (plans : PlanSelector) (p : Parameters)
    (mem : Memory) (m n c : Nat) : carryForm (contract (tensorValue plans p mem m n c)) := by
  cases c with
  | zero => trivial
  | succ c => exact contractAdd_carryForm _ _

private theorem replace_live_carry (cfg : FP.Config) (a b : Expr) (h : carryForm a) :
    evaluate cfg (contractAdd a b) =
      evaluate cfg (contractAdd (.word (evaluate cfg a)) b) := by
  cases a <;> cases b <;> simp_all [carryForm, contractAdd, evaluate]

private theorem read_wordLeaves (mem : ValueProofs.Memory) (r : RegionName) (o : Nat) :
    wordLeaves ((Native.liftMemory mem r o).read .real) = .word ((mem r o).read .real) := by
  unfold Native.liftMemory
  cases h : mem r o with
  | mk d v => cases d <;> simp [Native.mapCell, Native.mapValue, Cell.read, defaultValue, wordLeaves] <;> rfl

def inputsA (p : Parameters) (mem : ValueProofs.Memory) (m c : Nat) : Fin p.BK → ValueProofs.Word :=
  fun i => (mem p.A (m*p.K+(c*p.BK+i.val))).read .real

def inputsB (p : Parameters) (mem : ValueProofs.Memory) (m n c : Nat) : Fin p.BK → ValueProofs.Word :=
  fun i => (mem p.B (m*p.N*p.K+n*p.K+(c*p.BK+i.val))).read .real

def selectedPlan (plans : PlanSelector) (p : Parameters) (mem : ValueProofs.Memory)
    (m n c : Nat) : ReductionPlan p.BK :=
  plans p.BK (fun i => .mul
    ((Native.liftMemory mem p.A (m*p.K+(c*p.BK+i.val))).read .real)
    ((Native.liftMemory mem p.B (m*p.N*p.K+n*p.K+(c*p.BK+i.val))).read .real))

private theorem wordLeaves_tree (mem : ValueProofs.Memory) (p : Parameters) (m n c : Nat)
    (tree : ReductionTree p.BK) :
    wordLeaves (treeValue (fun i : Fin p.BK => .mul
      ((Native.liftMemory mem p.A (m*p.K+(c*p.BK+i.val))).read .real)
      ((Native.liftMemory mem p.B (m*p.N*p.K+n*p.K+(c*p.BK+i.val))).read .real)) tree) =
      productTree (inputsA p mem m c) (inputsB p mem m n c) tree := by
  induction tree with
  | input i => simp [treeValue, wordLeaves, read_wordLeaves, productTree, inputsA, inputsB]
  | zero => rfl
  | add x y hx hy => simp [treeValue, wordLeaves, productTree, hx, hy]

/-- Independent local numerical fold: exact source operands and live carry,
with no source execution or output-memory lookup. -/
def fold (D : DotAcc) (plans : PlanSelector) (p : Parameters)
    (mem : ValueProofs.Memory) (m n : Nat) : Nat → ValueProofs.Word
  | 0 => FP.Value.zero .fp32
  | c+1 => D p.BK (selectedPlan plans p mem m n c)
      (inputsA p mem m c) (inputsB p mem m n c) (fold D plans p mem m n c)

theorem graph_eq_local_fold (cfg : FP.Config) (D : DotAcc) (law : PrimitiveLaw cfg D)
    (plans : PlanSelector) (p : Parameters) (mem : ValueProofs.Memory) (m n c : Nat) :
    evaluate cfg (contract (tensorValue plans p (Native.liftMemory mem) m n c)) =
      fold D plans p mem m n c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    rw [tensorValue, contract, replace_live_carry _ _ _ (tensor_carryForm plans p _ m n c), ih]
    rw [fold, law]
    unfold dotAcc
    rw [← evaluate_wordLeaves cfg (contractAdd _ _)]
    rw [wordLeaves_contractAdd, wordLeaves_contract]
    change evaluate cfg (contractAdd (.word (fold D plans p mem m n c))
      (contract (wordLeaves (tensorBlock plans p (Native.liftMemory mem) m n c)))) = _
    unfold tensorBlock
    rw [wordLeaves_tree]
    rfl

/-- Composition through the already-proved original wrapper. The sole added
law concerns one local reduction invocation, never a whole output or run. -/
theorem returned_tensor_local_correct (allocate : Allocator) (rest : Algebra Expr)
    (cfg : FP.Config) (D : DotAcc) (law : PrimitiveLaw cfg D) (plans : PlanSelector)
    (a : Arguments) (initial : State ValueProofs.Word)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l) :
    match prepare allocate a initial.mem with
    | .error e => Native.runTensor allocate rest cfg plans a initial = .error e
    | .ok l => ∃ result : ReturnedTensor,
        Native.runTensor allocate rest cfg plans a initial = .ok (some result) ∧
        result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
        (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
          result.memory result.region (m.val*l.parameters.N+n.val) = Cell.mk .real
            (fold D plans l.parameters l.memory m.val n.val (l.parameters.K/l.parameters.BK))) ∧
        (∀ r o, ValueProofs.GridOutside l.parameters r o → result.memory r o = l.memory r o) := by
  have h := Native.returned_tensor_correct allocate rest cfg plans a initial abi
  cases hp : prepare allocate a initial.mem with
  | error e => simpa [hp] using h
  | ok l =>
    simp only [hp] at h ⊢
    obtain ⟨result,he,hr,hc,hregion,hvalue,hframe⟩ := h
    refine ⟨result,he,hr,hc,hregion,?_,hframe⟩
    intro m n
    rw [hvalue]
    unfold Native.expected
    rw [graph_eq_local_fold cfg D law]

#specNonCircular dotAcc avoiding [Native.runTensor, Native.expected, source, FP.Structural.exec]
#specNonCircular ResultLaw avoiding [Native.runTensor, Native.expected, source, FP.Structural.exec]
#specNonCircular fold avoiding [Native.runTensor, Native.expected, source, FP.Structural.exec]
#axiomsClean graph_eq_local_fold
#axiomsClean returned_tensor_local_correct
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph.Local
