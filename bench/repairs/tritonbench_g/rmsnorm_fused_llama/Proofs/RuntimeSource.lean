import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelSource
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeReference
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeCandidate
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.EpsilonABI
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeSource
open VeriTile Triton FP.Structural Contract KernelAPI
noncomputable section
set_option maxHeartbeats 5000000
variable {x w out : TensorInfo} {eps : Epsilon}

def epsilonTerm (word : Native.Word) : Native.Term := .leaf "epsilon" 0 word

theorem epsilon_shape (u v : Native.Word) : Native.shape (epsilonTerm u)=Native.shape (epsilonTerm v) := rfl

theorem epsilon_observation (I : Native.Instructions) (S : Native.Schedule) (word : Native.Word) :
    Native.eval I S (epsilonTerm word)=word := rfl

def nativeTarget (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (read : Heap Native.Word) (b : Binding x w out) (q : Host.Request) (epsilon : Native.Word) (row col : Nat) : Native.Word :=
  RuntimeReference.native I R S (fun j=>KernelSource.tensorGraph read b.input (row*q.columns+j))
    (KernelSource.tensorGraph read b.weight) q.columns 16384 (epsilonTerm epsilon) col

theorem native_result (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (read : Heap Native.Word) (b : Binding x w out) (ok : Admitted x w eps) (epsilon : Native.Word) (u : Nat → Native.Word) :
    let q := request x w eps ok
    ∃ final,RuntimeGrid.run (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 (epsilonTerm epsilon) (KernelSource.entry read b u) q.rows=some final ∧
      RuntimeCandidate.grid (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 (epsilonTerm epsilon) (KernelSource.entry read b u) q.rows=some final ∧
      (∀ row col,row<q.rows → col<q.columns →
        Native.observe I S final "y" (row*q.columns+col)=nativeTarget I R S read b q epsilon row col) ∧
      RuntimeGrid.Frame "y" q.size (KernelSource.entry read b u) final := by
  let q := request x w eps ok
  obtain ⟨final,ran,mem⟩ := RuntimeGrid.complete (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 (epsilonTerm epsilon)
    (KernelSource.entry read b u) q.positive q.limit q.rows
  obtain ⟨values,frame⟩ := RuntimeGrid.graph_result (Native.algebra R) "x" "y" "w" q.columns 16384 (epsilonTerm epsilon)
    (KernelSource.entry read b u) q.limit (by decide) (by decide) q.rows
  have facts := binding_facts b ok
  refine ⟨final,ran,?_,?_,?_⟩
  · rw [RuntimeCandidate.grid_eq]
    exact ran
  · intro row col hr hc
    unfold Native.observe
    rw [mem,values row hr col hc,Cell.read_same]
    apply congrArg (Native.eval I S)
    apply RuntimeReference.source_value
    · intro j hj
      change KernelSource.sourceGraph read b.input (row*q.columns+j)=KernelSource.tensorGraph read b.input (row*q.columns+j)
      apply KernelSource.graph_coordinate read b.input b.inputLaws facts.1
      rw [facts.2.2.2.2.2.1]
      exact Index.active_bound q row j hr hj
    · intro j hj
      change KernelSource.sourceGraph read b.weight j=KernelSource.tensorGraph read b.weight j
      apply KernelSource.graph_coordinate read b.weight b.weightLaws facts.2.1
      rw [facts.2.2.2.2.2.2.1]
      exact hj
    · exact hc
  · intro r i outside
    rw [mem]
    exact frame r i outside

#axiomsClean native_result
#guard_msgs (drop info) in
#specNonCircular nativeTarget avoiding [runtimeKernel,KernelSource.entry,RuntimeRow.value,RuntimeGrid.run,RuntimeGrid.graph]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RuntimeSource
