import bench.repairs.tritonbench_g.token_attn_llama2.Host
import bench.repairs.tritonbench_g.token_attn_llama2.Integers
import bench.repairs.tritonbench_g.token_attn_llama2.Binding
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-- Fixed complete reduction plans; only finite physical writer choices remain. -/
def nativeAlgebras (profile : Native.Profile) (winner : Native.Winner) : Program → Algebra Native.Term :=
  fun pid => Native.algebra (profile.reductions (winner pid))

def nativeGraph (profile : Native.Profile) (winner : Native.Winner)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (s : State Native.Word)
    (b h t : Nat) : Native.Term :=
  score (nativeAlgebras profile winner (b,h,t/32)) (c.parameters profile.hostScale)
    (start c.grid (b,h,t/32) (initial c m (Native.injectBuffers c x) (Native.injectState s)))
    (lane c profile.hostScale t)

def nativeTarget (I : Native.Instructions) (profile : Native.Profile) (winner : Native.Winner)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (s : State Native.Word)
    (b h t : Nat) : Native.Word :=
  Native.eval I Native.defaultFusion profile.scaleWord (nativeGraph profile winner c m x s b h t)

/-- Source/control/grid/host closure before interpreting any floating opcode.
Initial leaves are the current caller buffer words, including exceptional FP32. -/
theorem host_native_coverage (profile : Native.Profile) (winner : Native.Winner)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (s : State Native.Word)
    (abi : ABI c) (dense : Dense c) (guard : Guard c) (data : Data c m)
    (p : Placement c) (flat : Fin p.arena → Cell Native.Word)
    (binding : FlatBinding c p m x flat) (compiler : CompilerBinding c p profile)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    Formed c m ∧ ∃ result,
      invoke (nativeAlgebras profile winner) c profile.hostScale abi work
        (initial c m (Native.injectBuffers c x) (Native.injectState s))=some result ∧
      result.returned=none ∧ result.output=abi.tensor .output ∧
      Result c m (nativeGraph profile winner c m x s)
        (initial c m (Native.injectBuffers c x) (Native.injectState s)) result.state := by
  have v := domain_valid c m dense guard data
  exact ⟨formed c m v,host_result_with (nativeAlgebras profile winner) c profile.hostScale m
    (Native.injectBuffers c x) (Native.injectState s) v abi work legal⟩

/-- All output words and the finite storage frame, with no numerical premises.
The meaning of native is the pinned source-derived ordered instruction graph. -/
theorem host_native (I : Native.Instructions) (profile : Native.Profile) (winner : Native.Winner)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (s : State Native.Word)
    (abi : ABI c) (dense : Dense c) (guard : Guard c) (data : Data c m)
    (p : Placement c) (flat : Fin p.arena → Cell Native.Word)
    (binding : FlatBinding c p m x flat) (compiler : CompilerBinding c p profile)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    Formed c m ∧ ∃ result,
      invoke (nativeAlgebras profile winner) c profile.hostScale abi work
        (initial c m (Native.injectBuffers c x) (Native.injectState s))=some result ∧
      result.returned=none ∧ result.output=abi.tensor .output ∧
      Result c m (nativeTarget I profile winner c m x s) (initial c m x s)
        (Native.projectState I Native.defaultFusion profile.scaleWord s result.state) ∧
      FlatResult c p m (nativeTarget I profile winner c m x s) flat
        (flatten p flat (Native.projectState I Native.defaultFusion profile.scaleWord s result.state)) := by
  obtain ⟨bounds,result,run,ret,out,values,frame⟩ := host_native_coverage profile winner c m x s abi dense guard data p flat binding compiler work legal
  have res : Result c m (nativeTarget I profile winner c m x s) (initial c m x s)
      (Native.projectState I Native.defaultFusion profile.scaleWord s result.state) := by
    refine ⟨?_,?_⟩
    · intro b hb h hh t ht
      change Native.projectCell I Native.defaultFusion profile.scaleWord
        (result.state.mem "Out" (h*c.outHead+(m.begin b+t)*c.outSeq))=_
      rw [values b hb h hh t ht]
      rfl
    · intro r o hu
      change Native.projectCell I Native.defaultFusion profile.scaleWord (result.state.mem r o)=_
      rw [frame r o hu]
      exact Native.initial_project I Native.defaultFusion profile.scaleWord c m x s r o
  exact ⟨bounds,result,run,ret,out,res,finite_result c p m x s _ flat binding (domain_valid c m dense guard data) _ res⟩

/-- Only local addRN/mulRN/fmaRN agreement is deferred; no reduction callback,
whole tensor equality, successful run, final memory or fixed writer is assumed. -/
theorem fpequiv (actual standard : Native.Instructions)
    (agreement : Native.InstructionAgreement actual standard)
    (profile : Native.Profile) (winner : Native.Winner)
    (c : Call) (m : Metadata c) (x : Buffers Native.Word c) (s : State Native.Word)
    (abi : ABI c) (dense : Dense c) (guard : Guard c) (data : Data c m)
    (p : Placement c) (flat : Fin p.arena → Cell Native.Word)
    (binding : FlatBinding c p m x flat) (compiler : CompilerBinding c p profile)
    (work : List Program) (legal : (programs c.grid).Perm work) :
    Formed c m ∧ ∃ result,
      invoke (nativeAlgebras profile winner) c profile.hostScale abi work
        (initial c m (Native.injectBuffers c x) (Native.injectState s))=some result ∧
      result.returned=none ∧ result.output=abi.tensor .output ∧
      Result c m (nativeTarget standard profile winner c m x s) (initial c m x s)
        (Native.projectState actual Native.defaultFusion profile.scaleWord s result.state) ∧
      FlatResult c p m (nativeTarget standard profile winner c m x s) flat
        (flatten p flat (Native.projectState actual Native.defaultFusion profile.scaleWord s result.state)) := by
  rw [Native.instruction_equality actual standard agreement]
  exact host_native standard profile winner c m x s abi dense guard data p flat binding compiler work legal

#axiomsClean host_native_coverage
#axiomsClean host_native
#axiomsClean fpequiv
#auditStmt host_native_coverage
#auditStmt host_native
#auditStmt fpequiv
#specNonCircular nativeTarget avoiding [FP.Structural.exec,invoke,NativeGrid.runPrograms]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
