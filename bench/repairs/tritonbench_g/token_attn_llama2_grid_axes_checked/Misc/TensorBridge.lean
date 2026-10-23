import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Correct
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.NativeReference

/-! Optional correspondence between Python checks and the kernel input contract. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.TensorBridge
open VeriTile.Triton FP.Structural Whole

theorem python_checks_iff_preconditions (t : TensorCall) (words : Nat → Word) :
    (∃ launch, checkInput t words=.ok launch) ↔ InputContract t words := checks_iff t words

#axiomsClean python_checks_iff_preconditions
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.TensorBridge

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.TensorBridge.Real
open VeriTile.Triton FP.Structural Whole
noncomputable section
open scoped BigOperators
open Classical
set_option maxHeartbeats 5000000

/-- Independent gathered score on the caller's original logical Tensors. -/
def reference (t : TensorCall) (words : Nat → Word) (before : Cells ℝ) (b h n : Nat) : ℝ :=
  (∑ d : Fin t.call.dim,logical before t.q (queryIndex t b h d.val)*
    logical before t.k (keyIndex t words b h n d.val))*idealScale t.call.dim

theorem formula_logical (t : TensorCall) (words : Nat → Word) (before : Cells ℝ)
    (contract : InputContract t words) (launch : active t)
    (b h n : Nat) (hb : b<t.call.batches) (hh : h<t.call.heads)
    (hn : n<(metadata t words).len b) :
    realFormula t.call (idealScale t.call.dim) (metadata t words) (buffers t before) b h n=
      reference t words before b h n := by
  unfold realFormula dotScore reference
  congr 1
  apply Finset.sum_congr rfl
  intro d _
  rw [query_tensor t words before contract launch b h hb hh d,
    key_tensor t words before contract launch b h n hb hh hn d]

def execute (t : TensorCall) (words : Nat → Word) (before : Cells ℝ)
    (rest : State ℝ) (work : List Program) : Option (Cells ℝ) :=
  if active t then
    (NativeGrid.runPrograms (fun _ => realAlgebra) (t.call.parameters (idealScale t.call.dim))
      t.call.grid work (initial t.call (metadata t words) (buffers t before) rest)).map
        (project t words before)
  else some before

/-- Complete quoted-source grid success, all logical output Tensor cells,
the raw typed state/frame, and the entire native caller-cell frame. The Real
projection uses ideal sqrt scaling and arbitrary Real payloads. -/
theorem token_attention_kernel_correct (t : TensorCall) (words : Nat → Word)
    (before : Cells ℝ) (rest : State ℝ) (contract : InputContract t words)
    (work : List Program) (order : (programs t.call.grid).Perm work) :
    (active t → Formed t.call (metadata t words) ∧
      t.call.batches<2^31 ∧ t.call.heads≤65535 ∧ (t.maximum+31)/32≤65535) ∧
    ∃ after, execute t words before rest work=some after ∧
      (active t → ∃ raw,
        NativeGrid.runPrograms (fun _ => realAlgebra) (t.call.parameters (idealScale t.call.dim))
          t.call.grid work (initial t.call (metadata t words) (buffers t before) rest)=some raw ∧
        after=project t words before raw ∧
        Result t.call (metadata t words) (reference t words before)
          (initial t.call (metadata t words) (buffers t before) rest) raw) ∧
      (∀ b<t.call.batches, ∀ h<t.call.heads, ∀ n<(metadata t words).len b,
        after (t.output.tensor.storage.address (t.output.tensor.physicalIndex
          (h*t.call.outputTokens+(metadata t words).begin b+n)))=
            Cell.mk .fp32 (reference t words before b h n)) ∧
      (∀ address, Exterior t words address → after address=before address) ∧
      (∀ o∈[t.q.observation,t.k.observation,t.locations.observation,t.starts.observation,t.lengths.observation],
        ∀ i<o.metadata.numel,after (o.metadata.dataPtr.toNat+4*i)=before (o.metadata.dataPtr.toNat+4*i)) := by
  refine ⟨fun launch => ⟨formed_indices t words contract launch,launch_axes t words contract launch⟩,?_⟩
  by_cases launch : active t
  · have domain := valid t words contract launch
    obtain ⟨raw,run,values,frame⟩ := Whole.complete (fun _ => realAlgebra) t.call (idealScale t.call.dim)
      (metadata t words) (buffers t before) rest domain work order
    have result : Result t.call (metadata t words) (reference t words before)
        (initial t.call (metadata t words) (buffers t before) rest) raw := by
      refine ⟨?_,frame⟩
      intro b hb h hh n hn
      rw [values b hb h hh n hn]
      congr 1
      exact (score_initial_real t.call (idealScale t.call.dim) (metadata t words)
        (buffers t before) rest domain b h n hb hn).trans
        (formula_logical t words before contract launch b h n hb hh hn)
    refine ⟨project t words before raw,by simp [execute,launch,run],
      fun _ => ⟨raw,run,rfl,result⟩,?_,project_frame t words before raw,?_⟩
    · intro b hb h hh n hn
      rw [←output_physical t words contract b h n hb hh hn,
        project_read t words before raw b h n hb hh hn]
      exact result.1 b hb h hh n hn
    · intro o ho i hi
      exact project_frame t words before raw _ (read_exterior t words contract launch o ho i hi)
  · refine ⟨before,by simp [execute,launch],fun h => False.elim (launch h),?_,fun _ _ => rfl,fun _ _ _ _ => rfl⟩
    intro b hb h hh n hn
    exact False.elim (launch (coordinate_active t words contract b n hb hn))

#specNonCircular reference avoiding [source,VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source,
  execute,NativeGrid.runPrograms]
#axiomsClean formula_logical
#axiomsClean token_attention_kernel_correct
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.TensorBridge.Real

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.TensorBridge.NativeApplication
open VeriTile.Triton FP.Structural Whole
open KernelNative
noncomputable section
open Classical
set_option maxHeartbeats 5000000

def termInitial (t : TensorCall) (before : Cells Word) (rest : State Word) : State Native.Term :=
  initial t.call (metadata t (wordRead before)) (Native.injectBuffers t.call (buffers t before))
    (Native.injectState rest)

def execute (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (t : TensorCall) (before : Cells Word)
    (rest : State Word) (work : List Program) : Option (Cells Word) :=
  if active t then
    (NativeGrid.runPrograms (algebras trees) (t.call.parameters (hostScale t.call.dim))
      t.call.grid work (termInitial t before rest)).map
        (fun raw => project t (wordRead before) before
          (Native.projectState I fusion (scaleWord t.call.dim) rest raw))
  else some before

theorem result_project (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (t : TensorCall) (before : Cells Word)
    (rest : State Word) (raw : State Native.Term)
    (result : Result t.call (metadata t (wordRead before)) (referenceTerm trees t before)
      (termInitial t before rest) raw) :
    Result t.call (metadata t (wordRead before)) (reference I trees fusion t before)
      (initial t.call (metadata t (wordRead before)) (buffers t before) rest)
      (Native.projectState I fusion (scaleWord t.call.dim) rest raw) := by
  constructor
  · intro b hb h hh n hn
    change Native.projectCell I fusion (scaleWord t.call.dim) (raw.mem "Out" _)=_
    rw [result.1 b hb h hh n hn]
    rfl
  · intro r o outside
    change Native.projectCell I fusion (scaleWord t.call.dim) (raw.mem r o)=_
    rw [result.2 r o outside]
    exact Native.initial_project I fusion (scaleWord t.call.dim) t.call
      (metadata t (wordRead before)) (buffers t before) rest r o

/-- Successful execution of the entire quoted source, independent ordered
native reference on actual caller Tensor words, raw typed and native frames.
The only numerical parameters are scalar add/mul/FMA instructions; reduction
trees and contraction schedules inspect expression shapes rather than values.
No finite-word, numerical-law, successful-run, or row-agreement premise occurs. -/
theorem token_attention_kernel_fp_equiv (I : Native.Instructions)
    (trees : Program → Native.Reductions) (fusion : Native.Schedule)
    (t : TensorCall) (before : Cells Word) (rest : State Word)
    (contract : InputContract t (wordRead before)) (work : List Program)
    (order : (programs t.call.grid).Perm work) :
    (active t → Formed t.call (metadata t (wordRead before)) ∧
      t.call.batches<2^31 ∧ t.call.heads≤65535 ∧ (t.maximum+31)/32≤65535) ∧
    ∃ after, execute I trees fusion t before rest work=some after ∧
      (active t → ∃ raw,
        NativeGrid.runPrograms (algebras trees) (t.call.parameters (hostScale t.call.dim))
          t.call.grid work (termInitial t before rest)=some raw ∧
        Result t.call (metadata t (wordRead before)) (referenceTerm trees t before)
          (termInitial t before rest) raw ∧
        Result t.call (metadata t (wordRead before)) (reference I trees fusion t before)
          (initial t.call (metadata t (wordRead before)) (buffers t before) rest)
          (Native.projectState I fusion (scaleWord t.call.dim) rest raw) ∧
        after=project t (wordRead before) before
          (Native.projectState I fusion (scaleWord t.call.dim) rest raw)) ∧
      (∀ b<t.call.batches, ∀ h<t.call.heads, ∀ n<(metadata t (wordRead before)).len b,
        after (t.output.tensor.storage.address (t.output.tensor.physicalIndex
          (h*t.call.outputTokens+(metadata t (wordRead before)).begin b+n)))=
            Cell.mk .fp32 (reference I trees fusion t before b h n)) ∧
      (∀ address, Exterior t (wordRead before) address → after address=before address) ∧
      (∀ o∈[t.q.observation,t.k.observation,t.locations.observation,t.starts.observation,t.lengths.observation],
        ∀ i<o.metadata.numel,after (o.metadata.dataPtr.toNat+4*i)=before (o.metadata.dataPtr.toNat+4*i)) := by
  refine ⟨fun launch => ⟨formed_indices t (wordRead before) contract launch,
    launch_axes t (wordRead before) contract launch⟩,?_⟩
  by_cases launch : active t
  · have domain := valid t (wordRead before) contract launch
    obtain ⟨raw,run,values,frame⟩ := Whole.complete (algebras trees) t.call (hostScale t.call.dim)
      (metadata t (wordRead before)) (Native.injectBuffers t.call (buffers t before))
      (Native.injectState rest) domain work order
    have result : Result t.call (metadata t (wordRead before)) (referenceTerm trees t before)
        (termInitial t before rest) raw := by
      refine ⟨?_,frame⟩
      intro b hb h hh n hn
      rw [values b hb h hh n hn]
      congr 1
      exact source_reference trees t before rest contract launch b h n hb hh hn
    have nativeResult := result_project I trees fusion t before rest raw result
    let native := Native.projectState I fusion (scaleWord t.call.dim) rest raw
    refine ⟨project t (wordRead before) before native,by simp [execute,launch,termInitial,run,native],
      fun _ => ⟨raw,run,result,nativeResult,rfl⟩,?_,
      project_frame t (wordRead before) before native,?_⟩
    · intro b hb h hh n hn
      rw [←output_physical t (wordRead before) contract b h n hb hh hn,
        project_read t (wordRead before) before native b h n hb hh hn]
      exact nativeResult.1 b hb h hh n hn
    · intro o ho i hi
      exact project_frame t (wordRead before) before native _
        (read_exterior t (wordRead before) contract launch o ho i hi)
  · refine ⟨before,by simp [execute,launch],fun h => False.elim (launch h),?_,
      fun _ _ => rfl,fun _ _ _ _ => rfl⟩
    intro b hb h hh n hn
    exact False.elim (launch (coordinate_active t (wordRead before) contract b n hb hn))

theorem empty_identity (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (t : TensorCall) (before : Cells Word)
    (rest : State Word) (work : List Program) (empty : ¬active t) :
    execute I trees fusion t before rest work=some before := by simp [execute,empty]

#axiomsClean result_project
#axiomsClean token_attention_kernel_fp_equiv
#axiomsClean empty_identity
#specNonCircular KernelNative.reference avoiding [source,
  VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source,NativeGrid.runPrograms,execute]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.TensorBridge.NativeApplication
