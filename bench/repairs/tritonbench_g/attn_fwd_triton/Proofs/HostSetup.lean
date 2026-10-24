import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Host
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.SetupCalls

/-! Ordered public API preparation. Each Boolean records the ordinary success
or failure of the named tensor/runtime call; this is a source-level contract,
not an implementation proof of Torch or the CUDA driver. Grid feasibility is
checked only after preparation, allocation, the empty fast path, and scales. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HostSetup
open NativeControl
noncomputable section
open scoped Classical

inductive Outcome (p : Parameters) (BD : Nat) where
  | failed (call : Call)
  | empty (h : p.batches*p.heads*p.context*p.headDim=0)
  | launch (nonempty : p.batches*p.heads*p.context*p.headDim≠0) (feasible : CudaLaunchable p BD)

def run (p : Parameters) (BD : Nat) (ok : Call → Bool) : Outcome p BD :=
  match calls ok beforeEmpty with
  | .error c => .failed c
  | .ok _ =>
    if he : p.batches*p.heads*p.context*p.headDim=0 then .empty he else
      match calls ok nonemptyChecks with
      | .error c => .failed c
      | .ok _ =>
        if hl : CudaLaunchable p BD then
          if ok .compileAndLaunch then .launch he hl else .failed .compileAndLaunch
        else .failed .compileAndLaunch

theorem empty_returns_before_launch (p : Parameters) (BD : Nat) (ok : Call → Bool)
    (hp : calls ok beforeEmpty=.ok ()) (he : p.batches*p.heads*p.context*p.headDim=0) :
    run p BD ok=.empty he := by simp [run,hp,he]

theorem infeasible_launch_fails (p : Parameters) (BD : Nat) (ok : Call → Bool)
    (hp : calls ok beforeEmpty=.ok ()) (he : p.batches*p.heads*p.context*p.headDim≠0)
    (hs : calls ok nonemptyChecks=.ok ()) (hl : ¬CudaLaunchable p BD) :
    run p BD ok=.failed .compileAndLaunch := by simp [run,hp,he,hs,hl]

theorem successful_setup (p : Parameters) (BD : Nat) (ok : Call → Bool)
    (hp : calls ok beforeEmpty=.ok ()) (he : p.batches*p.heads*p.context*p.headDim≠0)
    (hs : calls ok nonemptyChecks=.ok ()) (hl : CudaLaunchable p BD) (hc : ok .compileAndLaunch=Bool.true) :
    run p BD ok=.launch he hl := by simp [run,hp,he,hs,hl,hc]

/-- Exact control bounds are conclusions of the successful ordinary setup
outcome; they are not an extra logical-input precondition. -/
theorem launched_control (p : Parameters) (BD : Nat) (he) (hl : CudaLaunchable p BD) :
    (Outcome.launch he hl : Outcome p BD)=.launch he hl ∧
      p.context≤274877906816 ∧ p.batches*p.heads≤65535 ∧ BD≤1048576 :=
  ⟨rfl,context p BD hl,hl.gridY,hl.arange⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HostSetup

