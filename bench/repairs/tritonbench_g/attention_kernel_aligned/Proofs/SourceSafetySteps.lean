import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.SafetyFragments
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Eliminate the concrete offset-list binder before recursive safety rewriting. -/
theorem safeBlockPtr2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (region : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      safeOp M precision bounds s base ∧ safeOp M precision bounds s a ∧ safeOp M precision bounds s b := by
  rw [safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem pure_safe (A : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (pure : ∀ stmt∈code,∀ s : State α,stepSafe A bounds stmt s) (s : State α) :
    traceSafeList A bounds code s := by
  induction code generalizing s with
  | nil => exact traceSafeList_nil _ _ _
  | cons stmt code ih =>
    apply traceSafeList_cons_intro (pure stmt (by simp) s)
    intro t _
    exact ih (fun q hq => pure q (by simp [hq])) t

def prefixResult (P : Primitives α) (p : Parameters) (s : State α) : State α :=
  let t := s.setReg "start_m" .nat [] (fun _ => s.pids 0)
  let t := t.setReg "off_hz" .nat [] (fun _ => s.pids 1)
  let t := t.setReg "off_b" .nat [] (fun _ => s.pids 1/p.heads)
  let t := t.setReg "off_h" .nat [] (fun _ => s.pids 1%p.heads)
  let t := t.setReg "q_offset" .nat [] (fun _ => headBase p s)
  let t := t.setReg "k_offset" .nat [] (fun _ => kvBase p s)
  let t := t.setReg "v_offset" .nat [] (fun _ => valueBase p s)
  let t := t.setReg "o_offset" .nat [] (fun _ => outputBase p s)
  let t := t.setReg "Q_block_ptr" .blockPtr [p.blockM,p.head] (fun _ => qPointer p s)
  let t := t.setReg "K_block_ptr" .blockPtr [p.head,p.blockN] (fun _ => kPointer p s 0)
  let t := t.setReg "V_block_ptr" .blockPtr [p.blockN,p.head] (fun _ => vPointer p s 0)
  let t := t.setReg "m_i" .real [p.blockM] (initialAccumulator P p).maximum
  let t := t.setReg "l_i" .real [p.blockM] (initialAccumulator P p).denominator
  let t := t.setReg "acc" .real [p.blockM,p.head] (initialAccumulator P p).numerator
  let t := t.setReg "qk_scale" .real [] (fun _ => scaleValue P p)
  t

def middleResult (P : Primitives α) (p : Parameters) (s current : State α) : State α :=
  let t := current.setReg "q" .fp16 [p.blockM,p.head] (qScaled P p s)
  let t := t.setReg "lo" .nat [] (fun _ => 0)
  let t := t.setReg "hi" .nat [] (fun _ => p.length)
  let t := t.setReg "b_ptr_offsets_m" .nat [p.blockM] (fun i => i.1.val)
  let t := t.setReg "b_offset" .nat [] (fun _ => biasBase p s)
  t.setReg "b_ptr_offsets_n_1" .nat [p.blockN] (fun i => i.1.val%p.biasLast+p.biasLast)

theorem prefix_safe (A : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList (backend A P) bounds (queryPrefix p) s := by
  apply pure_safe
  intro stmt member current
  simp only [queryPrefix,ComputeKernel.surfaceBody,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [stepSafe_unfold,safeExpr_unfold,safeCompute_unfold]
  all_goals try rw [safeBlockPtr2]
  all_goals simp only [safeOp_unfold,true_and,and_true]

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
