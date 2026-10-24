import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelReference
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.KernelCandidate

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelSource
open VeriTile Triton FP.Structural Contract KernelAPI
noncomputable section
set_option maxHeartbeats 5000000

variable {x w out : TensorInfo} {eps : Epsilon}

def realEntry (read : Heap ℝ) (b : Binding x w out) : BlockState where
  mem := fun r i=>.real (if r="x" then sourceRead read b.input i
    else if r="w" then sourceRead read b.weight i else sourceRead read b.output i)
  regs := fun _ _ _=>none
  pids := fun _=>0

/-- Word leaves retain actual physical addresses. Aliased inputs read from
one common heap; no copied or synthesized input vector is assumed. -/
def sourceGraph (read : Heap Native.Word) (t : Torch.FP32.Tensor) (i : Nat) : Native.Term :=
  .leaf "physical" (t.storage.address (t.storageOffset+i)) (sourceRead read t i)

def tensorGraph (read : Heap Native.Word) (t : Torch.FP32.Tensor) (i : Nat) : Native.Term :=
  .leaf "physical" (t.storage.address (t.physicalIndex i)) (tensorRead read t i)

def entry (read : Heap Native.Word) (b : Binding x w out) (undefinedWeight : Nat → Native.Word) : State Native.Term where
  mem := fun r i=>.mk .real (if r="x" then sourceGraph read b.input i
    else if r="w" then sourceGraph read b.weight i else sourceGraph read b.output i)
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun _=>0
  undef := fun d _ i=>match d with | .real=>.leaf "inactive_weight" i (undefinedWeight i) | d=>defaultValue d

theorem graph_coordinate (read : Heap Native.Word) (t : Torch.FP32.Tensor) (laws : Torch.FP32.TensorLaws t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    sourceGraph read t i=tensorGraph read t i := by
  simp [sourceGraph,tensorGraph,sourceRead,tensorRead,laws.contiguous contiguous i hi]

theorem real_result (read : Heap ℝ) (b : Binding x w out) (ok : Admitted x w eps) :
    let q := request x w eps ok
    ∃ final,RealGrid.run "x" "y" "w" q.rows q.columns 16384 q.epsilon (realEntry read b) q.rows=some final ∧
      KernelCandidate.realGrid "x" "y" "w" q.rows q.columns 16384 q.epsilon (realEntry read b) q.rows=some final ∧
      (∀ row col,row<q.rows → col<q.columns →
        final.readMem "y" (row*q.columns+col)=
          mathematical (tensorRead read b.input) (tensorRead read b.weight) row q.columns q.epsilon col) ∧
      RealGrid.Frame "y" q.size (realEntry read b) final := by
  let q := request x w eps ok
  obtain ⟨final,ran,values,frame⟩ := RealGrid.complete "x" "y" "w" q.rows q.columns 16384 q.epsilon
    (realEntry read b) (by decide) (by decide) (by decide) q.rows
  refine ⟨final,ran,?_,?_,frame⟩
  · rw [KernelCandidate.real_grid_eq]
    exact ran
  · intro row col hr hc
    rw [values row hr col hc]
    change mathematical (sourceRead read b.input) (sourceRead read b.weight) row q.columns q.epsilon col=_
    exact KernelReference.mathematical_congr _ _ _ _ row q.columns q.epsilon
      (fun j hj=>input_read b ok read row j hr hj) (fun j hj=>weight_read b ok read j hj) col hc

def nativeTarget (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (read : Heap Native.Word) (b : Binding x w out) (q : Host.Request) (row col : Nat) : Native.Word :=
  KernelReference.native I R S (fun j=>tensorGraph read b.input (row*q.columns+j))
    (tensorGraph read b.weight) q.columns 16384 q.epsilon col

theorem native_result (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (read : Heap Native.Word) (b : Binding x w out) (ok : Admitted x w eps) (u : Nat → Native.Word) :
    let q := request x w eps ok
    ∃ final,Grid.run (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 q.epsilon (entry read b u) q.rows=some final ∧
      KernelCandidate.grid (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 q.epsilon (entry read b u) q.rows=some final ∧
      (∀ row col,row<q.rows → col<q.columns →
        Native.observe I S final "y" (row*q.columns+col)=nativeTarget I R S read b q row col) ∧
      Grid.Frame "y" q.size (entry read b u) final := by
  let q := request x w eps ok
  obtain ⟨final,ran,mem⟩ := Grid.complete (Native.algebra R) "x" "y" "w" q.rows q.columns 16384 q.epsilon
    (entry read b u) q.positive q.limit q.rows
  obtain ⟨values,frame⟩ := Grid.graph_result (Native.algebra R) "x" "y" "w" q.columns 16384 q.epsilon
    (entry read b u) q.limit (by decide) (by decide) q.rows
  have facts := binding_facts b ok
  refine ⟨final,ran,?_,?_,?_⟩
  · rw [KernelCandidate.grid_eq]
    exact ran
  · intro row col hr hc
    unfold Native.observe
    rw [mem,values row hr col hc,Cell.read_same]
    apply congrArg (Native.eval I S)
    apply KernelReference.source_value
    · intro j hj
      change sourceGraph read b.input (row*q.columns+j)=tensorGraph read b.input (row*q.columns+j)
      apply graph_coordinate read b.input b.inputLaws facts.1
      rw [facts.2.2.2.2.2.1]
      exact Index.active_bound q row j hr hj
    · intro j hj
      change sourceGraph read b.weight j=tensorGraph read b.weight j
      apply graph_coordinate read b.weight b.weightLaws facts.2.1
      rw [facts.2.2.2.2.2.2.1]
      exact hj
    · exact hc
  · intro r i outside
    rw [mem]
    exact frame r i outside

#axiomsClean real_result
#axiomsClean native_result
#guard_msgs (drop info) in
#specNonCircular nativeTarget avoiding [kernel,entry,Row.value,Grid.run,Grid.graph,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.KernelSource
