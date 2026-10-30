import bench.tritonbench_g.triton_attention.Proofs.IOExecution
import bench.tritonbench_g.triton_attention.Proofs.IOBounds
import bench.tritonbench_g.triton_attention.Proofs.ForwardAccessSafety
import bench.tritonbench_g.triton_attention.Proofs.BackwardAccessSafety

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 2000000

theorem real_invocation_safe (kind : Native.Kind) (c : Shape) (scale : ℝ) (pid : Grid.Program)
    (legal : Grid.Legal (Native.gridFor c kind) pid) (bounds : RegionName→Nat)
    (backed : SourceAccess.Bounds c bounds) (memory : Grid.Memory Erased.R) :
    (realInvocation kind c scale pid).Safe bounds memory := by
  unfold MaterializedInvocation.Safe Invocation.Safe
  rw [real_initial]
  cases kind with
  | forward=>
      exact SourceAccess.forward_safe Erased.algebra Erased.model c scale
        (Grid.entry (Grid.forwardGrid c) pid memory) legal bounds backed
  | preprocess=>
      exact SourceAccess.preprocess_kernel_safe Erased.algebra Erased.model c
        (Grid.entry (Grid.preprocessGrid c) pid memory) legal.1 bounds backed
  | backward=>
      exact SourceAccess.backward_safe Erased.algebra Erased.model c scale
        (Grid.entry (Grid.backwardGrid c) pid memory) legal.1 bounds backed

theorem native_invocation_safe (one : Bool) (kind : Native.Kind) (c : Shape) (scale : ℝ)
    (owners : TritonAttention.Whole001.Native.Owners) (pid : Grid.Program)
    (legal : Grid.Legal (Native.gridFor c kind) pid) (bounds : RegionName→Nat)
    (backed : SourceAccess.Bounds c bounds) (memory : Grid.Memory Native.Graph) :
    (nativeInvocation one kind c scale owners pid).Safe bounds memory := by
  unfold MaterializedInvocation.Safe Invocation.Safe
  rw [native_initial]
  cases kind with
  | forward=>
      exact SourceAccess.forward_safe (Native.algebraScalar one .forward) (Native.modelScalar one .forward)
        c scale (Grid.entry (Grid.forwardGrid c) pid (Native.Selected.embed memory)) legal bounds backed
  | preprocess=>
      exact SourceAccess.preprocess_kernel_safe (Native.algebraScalar one .preprocess) (Native.modelScalar one .preprocess)
        c (Grid.entry (Grid.preprocessGrid c) pid (Native.Selected.embed memory)) legal.1 bounds backed
  | backward=>
      exact SourceAccess.backward_safe (Native.algebraScalar one .backward) (Native.modelScalar one .backward)
        c scale (Grid.entry (Grid.backwardGrid c) pid (Native.Selected.embed memory)) legal.1 bounds backed

theorem grid_safety {α β : Type} [Inhabited β] (job : Grid.Program→MaterializedInvocation α β)
    (bounds : RegionName→Nat) (work : List Grid.Program)
    (safe : ∀ pid∈work,∀ memory,(job pid).Safe bounds memory) (memory : Grid.Memory α) :
    MaterializedScheduleSafe bounds (work.map job) memory := by
  induction work generalizing memory with
  | nil=>trivial
  | cons pid work ih=>
    exact ⟨safe pid (List.mem_cons_self ..) memory,
      fun next _=>ih (fun p member=>safe p (List.mem_cons_of_mem pid member)) next⟩

theorem real_safe (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (xs : Inputs ℝ c) (bounds : RegionName→Nat)
    (backed : (realIO c scale fw pw bw).InBounds bounds xs) (memory : Grid.Memory Erased.R) :
    MaterializedScheduleSafe bounds ((realIO c scale fw pw bw).schedule xs) memory := by
  have sizes := in_bounds c some (WithBot.unbotD 0) _ xs bounds backed
  change MaterializedScheduleSafe bounds (_++_++_) memory
  rw [materializedScheduleSafe_append,materializedScheduleSafe_append]
  refine ⟨⟨?_,?_⟩,?_⟩
  · exact grid_safety _ bounds fw (fun pid member=>real_invocation_safe .forward c scale pid
      (Grid.FullSchedule.legal _ _ hf pid member) bounds sizes) memory
  · intro next _
    exact grid_safety _ bounds pw (fun pid member=>real_invocation_safe .preprocess c scale pid
      (Grid.FullSchedule.legal _ _ hp pid member) bounds sizes) next
  · intro next _
    exact grid_safety _ bounds bw (fun pid member=>real_invocation_safe .backward c scale pid
      (Grid.FullSchedule.legal _ _ hb pid member) bounds sizes) next

theorem native_safe (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (xs : Inputs Native.Word c) (bounds : RegionName→Nat)
    (backed : (nativeIO scalar c owners I literal fw pw bw).InBounds bounds xs) (memory : Grid.Memory Native.Graph) :
    MaterializedScheduleSafe bounds ((nativeIO scalar c owners I literal fw pw bw).schedule xs) memory := by
  have sizes := in_bounds c Native.Graph.word (Native.Graph.eval I literal) _ xs bounds backed
  change MaterializedScheduleSafe bounds (_++_++_) memory
  rw [materializedScheduleSafe_append,materializedScheduleSafe_append]
  refine ⟨⟨?_,?_⟩,?_⟩
  · exact grid_safety _ bounds fw (fun pid member=>native_invocation_safe (Native.constexprScale scalar) .forward c
      (scalar.project scalar.raw) owners pid (Grid.FullSchedule.legal _ _ hf pid member) bounds sizes) memory
  · intro next _
    exact grid_safety _ bounds pw (fun pid member=>native_invocation_safe (Native.constexprScale scalar) .preprocess c
      (scalar.project scalar.raw) owners pid (Grid.FullSchedule.legal _ _ hp pid member) bounds sizes) next
  · intro next _
    exact grid_safety _ bounds bw (fun pid member=>native_invocation_safe (Native.constexprScale scalar) .backward c
      (scalar.project scalar.raw) owners pid (Grid.FullSchedule.legal _ _ hb pid member) bounds sizes) next

#axiomsClean real_safe
#axiomsClean native_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
