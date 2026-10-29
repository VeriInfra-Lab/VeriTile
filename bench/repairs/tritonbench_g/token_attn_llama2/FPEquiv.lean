import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.Checked
import bench.repairs.tritonbench_g.token_attn_llama2.Proofs.FloatingPoint

namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 4000000

/-- Exact bounded host evaluation, followed by the FP32 ABI conversion. -/
def hostScale (D : Nat) : ℝ :=
  if D=16 then 1/4 else if D=32 then 1592262918131443/9007199254740992
  else if D=64 then 1/8 else 1592262918131443/18014398509481984

def scaleWord (D : Nat) : Native.Word :=
  if D=16 then 0x3e800000 else if D=32 then 0x3e3504f3
  else if D=64 then 0x3e000000 else 0x3db504f3

/-- A tree and contraction schedule describe compiler structure only. Every
reduction leaf is covered, and no callback can return a scalar result. The
historical pinned profiles remain available with their exact Binder domain. -/
def graphAlgebras (trees : Program → Native.Reductions) : Program → Algebra Native.Term :=
  fun pid => Native.algebra (trees pid)

def graph (trees : Program → Native.Reductions) (c : Call) (m : Metadata c)
    (x : Buffers Native.Word c) (s : State Native.Word) (b h t : Nat) : Native.Term :=
  score (graphAlgebras trees (b,h,t/32)) (c.parameters (hostScale c.dim))
    (start c.grid (b,h,t/32) (initial c m (Native.injectBuffers c x) (Native.injectState s)))
    (lane c (hostScale c.dim) t)

def fpTarget (I : Native.Instructions) (trees : Program → Native.Reductions) (fusion : Native.Schedule)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (s : State Native.Word) (b h t : Nat) : Native.Word :=
  Native.eval I fusion (scaleWord c.dim) (graph trees c m x s b h t)

/-- Independent rounded target from initial Q/K words, gathered metadata,
explicit ordered dimension trees, source mul/add/FMA and final scale multiply.
There is no Real execution dependency, whole reduction equality or run premise. -/
theorem checked_native (I : Native.Instructions) (trees : Program → Native.Reductions)
    (fusion : Native.Schedule) (c : Call) (m : Metadata c) (r : Request c)
    (accepted : check c m r=Bool.true) (x : Buffers Native.Word c) (s : State Native.Word)
    (flat : Fin (2^62) → Cell Native.Word)
    (binding : FlatBinding c (checkedPlacement c m r ((check_iff c m r).mp accepted)) m x flat)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    let a := (check_iff c m r).mp accepted
    Formed c m ∧ ∃ result,
      invoke (graphAlgebras trees) c (hostScale c.dim) (checkedABI c m r a) work
        (initial c m (Native.injectBuffers c x) (Native.injectState s))=some result ∧
      result.returned=none ∧ result.output=r.tensor .output ∧
      Result c m (fpTarget I trees fusion c m x s) (initial c m x s)
        (Native.projectState I fusion (scaleWord c.dim) s result.state) ∧
      FlatResult c (checkedPlacement c m r a) m (fpTarget I trees fusion c m x s) flat
        (flatten (checkedPlacement c m r a) flat (Native.projectState I fusion (scaleWord c.dim) s result.state)) := by
  let a := (check_iff c m r).mp accepted
  have v := checked_valid c m r a
  obtain ⟨result,run,ret,out,values,frame⟩ := host_result_with (graphAlgebras trees) c
    (hostScale c.dim) m (Native.injectBuffers c x) (Native.injectState s) v
    (checkedABI c m r a) work legal
  have res : Result c m (fpTarget I trees fusion c m x s) (initial c m x s)
      (Native.projectState I fusion (scaleWord c.dim) s result.state) := by
    refine ⟨?_,?_⟩
    · intro b hb h hh t ht
      change Native.projectCell I fusion (scaleWord c.dim)
        (result.state.mem "Out" (h*c.outHead+(m.begin b+t)*c.outSeq))=_
      rw [values b hb h hh t ht]
      rfl
    · intro region offset miss
      change Native.projectCell I fusion (scaleWord c.dim) (result.state.mem region offset)=_
      rw [frame region offset miss]
      exact Native.initial_project I fusion (scaleWord c.dim) c m x s region offset
  exact ⟨formed c m v,result,run,ret,out,res,
    finite_result c (checkedPlacement c m r a) m x s _ flat binding v _ res⟩

/-- The deferred numerical boundary consists only of rounded FP32 add, mul
and FMA word semantics. This is not an exact-Real interpretation. -/
theorem checked_fpequiv (actual standard : Native.Instructions)
    (agreement : Native.InstructionAgreement actual standard)
    (trees : Program → Native.Reductions) (fusion : Native.Schedule)
    (c : Call) (m : Metadata c) (r : Request c) (accepted : check c m r=Bool.true)
    (x : Buffers Native.Word c) (s : State Native.Word) (flat : Fin (2^62) → Cell Native.Word)
    (binding : FlatBinding c (checkedPlacement c m r ((check_iff c m r).mp accepted)) m x flat)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    let a := (check_iff c m r).mp accepted
    Formed c m ∧ ∃ result,
      invoke (graphAlgebras trees) c (hostScale c.dim) (checkedABI c m r a) work
        (initial c m (Native.injectBuffers c x) (Native.injectState s))=some result ∧
      result.returned=none ∧ result.output=r.tensor .output ∧
      Result c m (fpTarget standard trees fusion c m x s) (initial c m x s)
        (Native.projectState actual fusion (scaleWord c.dim) s result.state) ∧
      FlatResult c (checkedPlacement c m r a) m (fpTarget standard trees fusion c m x s) flat
        (flatten (checkedPlacement c m r a) flat (Native.projectState actual fusion (scaleWord c.dim) s result.state)) := by
  rw [Native.instruction_equality actual standard agreement]
  exact checked_native standard trees fusion c m r accepted x s flat binding work legal

/-- Full checked FP dispatch. Empty calls keep their original descriptor and
current state without requiring canonical inactive strides or a typed heap. -/
theorem checked_fp_call (actual standard : Native.Instructions)
    (agreement : Native.InstructionAgreement actual standard)
    (trees : Program → Native.Reductions) (fusion : Native.Schedule)
    (c : Call) (m : Metadata c) (r : Request c) (launch : Bool)
    (accepted : validate c m r=.ok launch)
    (x : Buffers Native.Word c) (s : State Native.Word) (flat : Fin (2^62) → Cell Native.Word)
    (binding : ∀ a : Admission c m r, FlatBinding c (checkedPlacement c m r a) m x flat)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    ∃ result, dispatch (graphAlgebras trees) c m r (hostScale c.dim) work
        (initial c m (Native.injectBuffers c x) (Native.injectState s))=.ok (some result) ∧
      result.returned=none ∧ result.output=r.tensor .output ∧
      ((launch=Bool.false ∧ result.state=initial c m (Native.injectBuffers c x) (Native.injectState s)) ∨
       (∃ a : Admission c m r, launch=Bool.true ∧ Formed c m ∧
        Result c m (fpTarget standard trees fusion c m x s) (initial c m x s)
          (Native.projectState actual fusion (scaleWord c.dim) s result.state) ∧
        FlatResult c (checkedPlacement c m r a) m (fpTarget standard trees fusion c m x s) flat
          (flatten (checkedPlacement c m r a) flat (Native.projectState actual fusion (scaleWord c.dim) s result.state)))) := by
  cases launch
  · exact ⟨⟨initial c m (Native.injectBuffers c x) (Native.injectState s),r.tensor .output,none⟩,
      dispatch_empty _ _ _ _ _ _ _ accepted,rfl,rfl,Or.inl ⟨rfl,rfl⟩⟩
  · have a := ((validate_launch c m r).mp accepted).2
    have hc := (check_iff c m r).mpr a
    obtain ⟨formed,result,run,ret,out,res,frame⟩ := checked_fpequiv actual standard agreement trees fusion c m r hc x s flat (binding _) work legal
    refine ⟨result,?_,ret,out,Or.inr ⟨a,rfl,formed,res,frame⟩⟩
    rw [dispatch_launch _ _ _ _ _ _ _ accepted]
    exact congrArg Except.ok run

#axiomsClean checked_fp_call
#axiomsClean checked_native
#axiomsClean checked_fpequiv
#auditStmt checked_native
#auditStmt checked_fpequiv
#specNonCircular fpTarget avoiding [FP.Structural.exec,invoke,NativeGrid.runPrograms]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
