import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelExecution
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-! Independent per-lane mathematics: inclusive forward prefixes and reverse
prefix sums inside the actual chunk, including its short final phase. The
source decimal constant is retained; no exact reciprocal-ln2 replacement. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelReal
open VeriTile.Triton FP.Structural Geometry
open scoped BigOperators
noncomputable section

def exp2 (x : ℝ) : ℝ := (2:ℝ)^x
def forward (g : Nat → ℝ) (n : Nat) : ℝ :=
  ∑ i∈Finset.range n,g i*1.44269504
def prepareQ (scale q g : ℝ) : ℝ := q*(exp2 g*scale)
def prepareK (k last g : ℝ) : ℝ := k*exp2 (last-g)
def backwardQ (inner inter g : ℝ) : ℝ := inner+inter*exp2 g
def backwardK (inner inter last g : ℝ) : ℝ := inner+inter*exp2 (last-g)
def gradient (span : Nat) (read : RegionName → Nat → ℝ) (t : Nat) : ℝ :=
  backwardQ (read "dqi" t) (read "dqo" t) (read "g" t)*read "q" t-
    backwardK (read "dki" t) (read "dko" t) (read "g" (span-1)) (read "g" t)*read "k" t
def backward (span : Nat) (read : RegionName → Nat → ℝ) (n : Nat) : ℝ :=
  ∑ i∈Finset.range n,gradient span read (span-1-i)
def evaluate (kind : Kind) (span : Nat) (scale : ℝ)
    (read : RegionName → Nat → ℝ) (r : RegionName) (t : Nat) : ℝ :=
  match kind with
  | .forward => forward (read "g") (t+1)
  | .prepare => if r="qg" then prepareQ scale (read "q" t) (read "g" t)
      else prepareK (read "k" t) (read "g" (span-1)) (read "g" t)
  | .backward => if r="dqo" then backwardQ (read "dqi" t) (read "dqo" t) (read "g" t)
      else if r="dko" then backwardK (read "dki" t) (read "dko" t) (read "g" (span-1)) (read "g" t)
      else backward span read (span-t)

def algebra (rest : Algebra ℝ) : Algebra ℝ :=
  { rest with
    literal := fun _ _ x => x
    binary := fun _ _ op a b => match op with
      | .add => a+b | .sub => a-b | .mul => a*b | _ => rest.binary none .real op a b
    unary := fun _ op a => match op with | .exp2 => exp2 a | _ => rest.unary none op a
    cast := fun _ _ _ x => x
    fp32Load := id }

theorem forward_value (rest : Algebra ℝ) (g : Nat → ℝ) (n : Nat) :
    FPReference.forwardCarry (algebra rest) g n=forward g n := by
  induction n with
  | zero => simp [FPReference.forwardCarry,FPReference.zero,algebra,forward]
  | succ n ih =>
    rw [FPReference.forwardCarry,ih]
    change forward g n+g n*1.44269504=forward g (n+1)
    simp only [forward,Finset.sum_range_succ]

theorem backward_value (rest : Algebra ℝ) (span : Nat) (read : RegionName → Nat → ℝ) (n : Nat) :
    FPReference.backwardCarry (algebra rest) span (read "dqi") (read "dqo")
      (read "dki") (read "dko") (read "q") (read "k") (read "g") n=backward span read n := by
  induction n with
  | zero => simp [FPReference.backwardCarry,FPReference.zero,algebra,backward]
  | succ n ih =>
    rw [FPReference.backwardCarry,ih]
    change backward span read n+gradient span read (span-1-n)=backward span read (n+1)
    simp only [backward,Finset.sum_range_succ]

theorem backward_stored (rest : Algebra ℝ) (r : RegionName) (x : ℝ) :
    FPReference.backwardStored (algebra rest) r x = .mk .real x := by
  simp [FPReference.backwardStored,FPReference.stored,algebra]

theorem evaluate_value (rest : Algebra ℝ) (kind : Kind) (span : Nat) (scale : ℝ)
    (read : RegionName → Nat → ℝ) (r : RegionName) (t : Nat) :
    FPLogical.evaluate (algebra rest) kind span scale read r t=.mk .real (evaluate kind span scale read r t) := by
  cases kind
  · simp only [FPLogical.evaluate,FPReference.forward,evaluate,forward_value]
    rfl
  · simp only [FPLogical.evaluate,evaluate]
    split <;> simp [FPReference.prepareQ,FPReference.prepareK,FPReference.stored,FPReference.mul,
      FPReference.sub,FPReference.exp2,FPReference.gate,algebra,prepareQ,prepareK]
  · simp only [FPLogical.evaluate,evaluate,backward_stored]
    split
    · simp [FPReference.backwardQ,FPReference.stored,FPReference.add,FPReference.mul,FPReference.exp2,
        FPReference.gate,algebra,backwardQ]
    · split
      · simp [FPReference.backwardK,FPReference.stored,FPReference.add,FPReference.mul,FPReference.sub,
          FPReference.exp2,FPReference.gate,algebra,backwardK]
      · rw [backward_value rest span read]

def Outputs (kind : Kind) (h : Call) (p : Phase) (prog : Program h p) (scale : ℝ)
    (u v : State ℝ) : Prop := ∀ r∈FPTile.outputs kind, ∀ (t : Fin p.span) (j : Fin h.BK),
  v.mem r (FPPrefix.off h p prog t.val j)=
    .mk .real (evaluate kind p.span scale (KernelExecution.read h p prog u j) r t.val)

theorem complete (rest : Algebra ℝ) (kind : Kind) (h : Call) (p : Phase)
    (hp : p∈phases h) (prog : Program h p) (scale : ℝ) (u : State ℝ)
    (hf : u.pids 0=prog.feature) (hc : u.pids 1=prog.chunk) (hr : u.pids 2=prog.row) :
    ∃ v,KernelExecution.Facts (algebra rest) kind h p prog scale u v ∧ Outputs kind h p prog scale u v := by
  obtain ⟨v,facts⟩ := KernelExecution.phase_complete (algebra rest) kind h p hp prog scale u hf hc hr
  refine ⟨v,facts,?_⟩
  intro r member t j
  rw [facts.outputs r member t j,evaluate_value]

#axiomsClean complete
#specNonCircular evaluate avoiding [Kernels.source,FP.Structural.exec,FPLogical.evaluate]
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelReal
