import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.KernelBinding
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteExecution
import bench.tritonbench_g.batched_vecmat_mult.Proofs.NativeContract
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Real

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.KernelModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open KernelDomain KernelBinding
noncomputable section
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩

/-- The actual initial kernel memory binds both operands to logical elements
of the actual allocated Tensors. All remaining typed memory is arbitrary. -/
def inputMemory (xs ys : Nat → ValueProofs.Word) (scratch : ValueProofs.Memory) : ValueProofs.Memory :=
  fun r i => if r="a" then .mk .real (xs i) else if r="b" then .mk .real (ys i) else scratch r i

def initial (xs ys : Nat → ValueProofs.Word) (scratch : State ValueProofs.Word) : State SourceGraph.Expr :=
  SourceGraph.Native.liftState scratch (inputMemory xs ys scratch.mem)

/-- Operand-only reference: it cannot inspect output memory or execute source. -/
def operandMemory (xs ys : Nat → ValueProofs.Word) : ValueProofs.Memory :=
  inputMemory xs ys (fun _ _ => .mk .real (FP.Value.zero .fp32))

/-- Independent full-K mathematical formula. Real denotes mathematical
arithmetic under the chosen interpretation of input words, not native rounding. -/
def mathematical (decode : ValueProofs.Word → ℝ) (a : Arguments)
    (xs ys : Nat → ValueProofs.Word) (m n : Nat) : ℝ :=
  ∑ k : Fin a.K,decode (xs (m*a.K+k.val))*decode (ys (m*a.N*a.K+n*a.K+k.val))

/-- Independent native operation graph. A selected valid reduction tree and
the explicit contraction pass determine FP.mul/add/fma evaluations. This
neither assumes a local dot oracle nor equates different floating trees. -/
def nativeTarget (cfg : FP.Config) (plans : SourceGraph.PlanSelector) (a : Arguments)
    (xs ys : Nat → ValueProofs.Word) (m n : Nat) : ValueProofs.Word :=
  SourceGraph.Native.expected cfg plans (parameters a) (operandMemory xs ys) m n

def nativeValues (cfg : FP.Config) (final : State SourceGraph.Expr) (i : Nat) : ValueProofs.Word :=
  ((SourceGraph.Native.decodeMemory cfg final.mem) "out" i).read .real

def realValues (decode : ValueProofs.Word → ℝ) (final : State SourceGraph.Expr) (i : Nat) : ℝ :=
  SourceGraph.realValue decode (SourceGraph.contract ((final.mem "out" i).read .real))

theorem tensor_initial (plans : SourceGraph.PlanSelector) (a : Arguments)
    (xs ys : Nat → ValueProofs.Word) (scratch : State ValueProofs.Word) (m n c : Nat) :
    SourceGraph.tensorValue plans (parameters a) (initial xs ys scratch).mem m n c=
      SourceGraph.tensorValue plans (parameters a)
        (SourceGraph.Native.liftMemory (operandMemory xs ys)) m n c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    rw [SourceGraph.tensorValue,SourceGraph.tensorValue,ih]
    congr 1

theorem mathematical_initial (decode : ValueProofs.Word → ℝ) (a : Arguments)
    (xs ys : Nat → ValueProofs.Word) (scratch : State ValueProofs.Word) (m n : Nat) :
    RealProofs.tensorExpected decode (parameters a) (initial xs ys scratch).mem m n=
      mathematical decode a xs ys m n := by
  rfl

theorem flat_inside (a : Arguments) (m : Fin a.M) (n : Fin a.N) : m.val*a.N+n.val<a.M*a.N := by
  have h := Nat.mul_le_mul_right a.N (Nat.succ_le_of_lt m.isLt)
  rw [Nat.succ_mul] at h
  have := n.isLt
  omega

#specNonCircular mathematical avoiding [source,initial,FP.Structural.exec,SourceGraph.runGrid,nativeValues,realValues]
#specNonCircular nativeTarget avoiding [source,initial,FP.Structural.exec,SourceGraph.runGrid,nativeValues,realValues]
#axiomsClean tensor_initial
#axiomsClean mathematical_initial
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.KernelModel
