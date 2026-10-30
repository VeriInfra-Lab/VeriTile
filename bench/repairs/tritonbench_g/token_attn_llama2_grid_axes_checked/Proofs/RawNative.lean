import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Kernels
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.NativeKernel

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.RawNative
open VeriTile.Triton FP.Structural Whole KernelNative
noncomputable section

/-- Raw load/dot/scale recurrence. It never executes the source program. -/
def referenceTerm (trees : Program → Native.Reductions) (c : Call) (m : Metadata c)
    (x : Buffers Native.Word c) (rest : State Native.Word) (b h n : Nat) : Native.Term :=
  VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.score (algebras trees (b,h,n/32))
    (c.parameters (hostScale c.dim))
    (start c.grid (b,h,n/32) (initial c m (Native.injectBuffers c x) (Native.injectState rest)))
    (lane c (hostScale c.dim) n)

def reference (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (c : Call) (m : Metadata c) (x : Buffers Native.Word c)
    (rest : State Native.Word) (b h n : Nat) : Native.Word :=
  Native.eval I fusion (scaleWord c.dim) (referenceTerm trees c m x rest b h n)

/-- Complete source graph execution, every independent native word, and both
the raw graph and projected native typed-memory frames. The original fixed
scale words, primitive instructions, reduction plans and fusion policy remain
unchanged and accept unrestricted payload words. -/
theorem token_attention_kernel_fp_equiv (I : Native.Instructions)
    (trees : Program → Native.Reductions) (fusion : Native.Schedule)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (rest : State Native.Word)
    (domain : Valid c m) (work : List Program) (order : (programs c.grid).Perm work) :
    ∃ raw, NativeGrid.runPrograms (algebras trees) (c.parameters (hostScale c.dim))
      c.grid work (initial c m (Native.injectBuffers c x) (Native.injectState rest))=some raw ∧
      Result c m (referenceTerm trees c m x rest)
        (initial c m (Native.injectBuffers c x) (Native.injectState rest)) raw ∧
      Result c m (reference I trees fusion c m x rest) (initial c m x rest)
        (Native.projectState I fusion (scaleWord c.dim) rest raw) := by
  obtain ⟨raw,run,result⟩ := Whole.complete (algebras trees) c (hostScale c.dim)
    m (Native.injectBuffers c x) (Native.injectState rest) domain work order
  refine ⟨raw,run,result,?_,?_⟩
  · intro b hb h hh n hn
    change Native.projectCell I fusion (scaleWord c.dim) (raw.mem "Out" _)=_
    rw [result.1 b hb h hh n hn]
    rfl
  · intro r o outside
    change Native.projectCell I fusion (scaleWord c.dim) (raw.mem r o)=_
    rw [result.2 r o outside]
    exact Native.initial_project I fusion (scaleWord c.dim) c m x rest r o

/-- An empty raw launch preserves all typed cells without metadata admission. -/
theorem empty_identity (A : Program → Algebra α) (c : Call) (scale : ℝ)
    [Inhabited α] (s : State α) :
    NativeGrid.runPrograms A (c.parameters scale) c.grid [] s = some s := rfl

#axiomsClean token_attention_kernel_fp_equiv
#axiomsClean empty_identity
#specNonCircular reference avoiding [source,NativeGrid.runPrograms]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.RawNative
