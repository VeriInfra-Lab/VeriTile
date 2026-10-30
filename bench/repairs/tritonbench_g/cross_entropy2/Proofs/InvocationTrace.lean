import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Invocation

namespace VeriTile.Bench.Repairs.CrossEntropy2.Invocation
open VeriTile.Triton Source Host Floating
open Floating.Profiled (DeviceFamily BackendABI BackwardABI BackwardOrder DeviceSelected TorchSelected BackwardSelected)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

def TorchCalls {σ : Type} (layout : Layout σ) (abi : BackendABI) (host : Profile)
    (rt : Runtime Compiled σ) (count : Nat) (loss lse : Tensor) (before : World Compiled σ) : Prop :=
  count≤1 ∨ ∃ (localLSE localLoss : Tensor) (logged summed : World Compiled σ),
    rt.logsumexp0 lse before=.ok localLSE logged ∧ rt.sum0 loss logged=.ok localLoss summed ∧
    ∀ positive : 0<count,
      host.hostMaxTree count positive=abi.torchLSEMax (Floating.Profiled.torchMetadata layout before lse) count positive ∧
      host.hostLSESumTree count positive=abi.torchLSESum (Floating.Profiled.torchMetadata layout before lse) count positive ∧
      host.hostSumTree count positive=abi.torchSum (Floating.Profiled.torchMetadata layout logged loss) count positive

theorem aggregate_success {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ)
    (o : Options) (count : Nat) (labels loss lse : Tensor) (before after : World Compiled σ)
    (host : Profile) (returned : Return)
    (call : aggregate layout abi rt o count labels loss lse before=.ok (host,returned) after) :
    Host.aggregate rt o count labels loss lse before=.ok returned after ∧
    TorchCalls layout abi host rt count loss lse before := by
  unfold aggregate at call
  obtain ⟨chosen,reduced,localCall,tail⟩ := (bind_success _ _ _ _).mp call
  by_cases many : count>1
  · simp only [if_pos many] at localCall
    obtain ⟨localLSE,logged,lseCall,rest⟩ := (bind_success _ _ _ _).mp localCall
    obtain ⟨localLoss,summed,sumCall,done⟩ := (bind_success _ _ _ _).mp rest
    cases done
    obtain ⟨hostEq,finished⟩ := (mark_success _ _ _ _ _).mp tail
    subst host
    refine ⟨?_,Or.inr ⟨localLSE,localLoss,logged,_,lseCall,sumCall,fun _ => ⟨rfl,rfl,rfl⟩⟩⟩
    simpa only [Host.aggregate,if_pos many,lseCall,sumCall,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind,finish] using finished
  · simp only [if_neg many] at localCall
    cases localCall
    obtain ⟨hostEq,finished⟩ := (mark_success _ _ _ _ _).mp tail
    subst host
    refine ⟨?_,Or.inl (by omega)⟩
    simpa only [Host.aggregate,if_neg many,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind,finish] using finished

theorem forward_launch_success {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ)
    (p : Parameters) (count : Nat) (before after : World Compiled σ) (device : DeviceFamily)
    (call : forwardLaunch layout abi rt p count before=.ok device after) :
    Host.forwardLaunch (Floating.Profiled.algebra device p.logitScale) rt p count before=.ok () after ∧
    (p.rows=0 ∨ ∃ ready,
      rt.setupForward2 p count before=.ok () ready ∧
      after={ready with memory:=Grid.forward (Floating.Profiled.algebra device p.logitScale) p ready.memory count} ∧
      ∀ positive : 0<p.block,
        device p.block positive=abi.device (Floating.Scheduled.keyFor layout ready p) p.block positive) := by
  by_cases empty : p.rows=0
  · simp only [forwardLaunch,if_pos empty] at call
    cases call
    exact ⟨by simp only [Host.forwardLaunch,if_pos empty],Or.inl empty⟩
  · simp only [forwardLaunch,if_neg empty] at call
    obtain ⟨u,ready,setup,done⟩ := (bind_success _ _ _ _).mp call
    cases u
    cases done
    exact ⟨by simp only [Host.forwardLaunch,if_neg empty,setup,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind],
      Or.inr ⟨ready,setup,rfl,fun _ => rfl⟩⟩

theorem backward_launch_success {σ : Type} (layout : Layout σ) (abi : BackwardABI) (rt : Runtime Compiled σ)
    (device : DeviceFamily) (p : Parameters) (count : Nat) (before after : World Compiled σ) (order : BackwardOrder)
    (call : backwardLaunch layout abi device rt p count before=.ok order after) :
    Host.backwardLaunch (Floating.Profiled.algebra device p.logitScale) rt p count before=.ok () after ∧
    (p.rows=0 ∨ ∃ ready,
      rt.setupBackward2 p count before=.ok () ready ∧
      after={ready with memory:=backwardGrid (Floating.Profiled.algebra device p.logitScale) p ready.memory count} ∧
      order=abi (Floating.Profiled.backwardKey layout ready p)) := by
  by_cases empty : p.rows=0
  · simp only [backwardLaunch,if_pos empty] at call
    cases call
    exact ⟨by simp only [Host.backwardLaunch,if_pos empty],Or.inl empty⟩
  · simp only [backwardLaunch,if_neg empty] at call
    obtain ⟨u,ready,setup,done⟩ := (bind_success _ _ _ _).mp call
    cases u
    cases done
    exact ⟨by simp only [Host.backwardLaunch,if_neg empty,setup,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind],Or.inr ⟨ready,setup,rfl,rfl⟩⟩

/-- Forgetting graph annotations preserves the very same successful source
calls, ready state, source parameters, returned objects and final state. -/
theorem forward_success {σ : Type} (layout : Layout σ) (abi : BackendABI) (rt : Runtime Compiled σ)
    (o : Options) (x y : Tensor) (rows cols : Nat) (initial final : World Compiled σ)
    (device : DeviceFamily) (host : Profile) (returned : Return)
    (shape : x.shape=[rows,cols])
    (success : forward layout abi rt o x y initial=.ok (device,host,returned) final) :
    ∃ e : ForwardData (Floating.Profiled.algebra device o.logitScale) rt o x y rows cols initial returned final,
      Host.forward (Floating.Profiled.algebra device o.logitScale) rt o x y initial=.ok returned final ∧
      DeviceSelected layout abi device e ∧ TorchSelected layout abi host e := by
  classical
  by_cases admitted : valid o x y rows cols
  · simp only [forward,shape,if_pos admitted] at success
    obtain ⟨logits,sx,hx,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨labels,sy,hy,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨loss,sl,hl,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨lse,se,he,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨zLoss,sz,hz,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨chosen,computed,hlaunch,hfinish⟩ := (bind_success _ _ _ _).mp success
    let count := VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width
    let split := o.split || decide (count>1 ∨ o.start≠0 ∨ cols≠o.total)
    let p := parameters o logits labels loss lse zLoss rows cols split
    change forwardLaunch layout abi rt p count sz=.ok chosen computed at hlaunch
    change (if split then mark chosen (aggregate layout abi rt o count labels loss lse computed)
      else .ok (chosen,witnessProfile,(loss,lse,zLoss)) computed)=.ok (device,host,returned) final at hfinish
    have rest : chosen=device ∧
        (if split then Host.aggregate rt o count labels loss lse computed else .ok (loss,lse,zLoss) computed)=.ok returned final ∧
        (split=Bool.false ∨ TorchCalls layout abi host rt count loss lse computed) := by
      cases hs : split with
      | «false» =>
        simp only [hs,Bool.false_eq_true,↓reduceIte] at hfinish
        cases hfinish
        exact ⟨rfl,by simp only [hs,Bool.false_eq_true,↓reduceIte],Or.inl rfl⟩
      | «true» =>
        simp only [hs,↓reduceIte] at hfinish
        obtain ⟨eq,agg⟩ := (mark_success _ _ _ _ _).mp hfinish
        have result := aggregate_success layout abi rt o count labels loss lse computed final host returned agg
        exact ⟨eq,by simpa only [hs,↓reduceIte] using result.1,Or.inr result.2⟩
    rcases rest with ⟨chosenEq,finished,torch⟩
    subst chosen
    have launch := forward_launch_success layout abi rt p count sz computed device hlaunch
    let e : ForwardData (Floating.Profiled.algebra device o.logitScale) rt o x y rows cols initial returned final :=
      ⟨admitted,logits,labels,sx,sy,hx,hy,loss,lse,zLoss,sl,se,sz,computed,count,rfl,split,rfl,hl,he,hz,launch.1,finished⟩
    refine ⟨e,?_,launch.2,torch⟩
    have originalLaunch : Host.forwardLaunch (Floating.Profiled.algebra device o.logitScale) rt
        (parameters o logits labels loss lse zLoss rows cols split) count sz=.ok () computed := launch.1
    dsimp only [count,split] at originalLaunch
    simpa only [Host.forward,shape,if_pos admitted,hx,hy,hl,he,hz,originalLaunch,
      VeriTile.Bench.Repairs.CrossEntropy1.Host.bind,count,split] using finished
  · simp [forward,shape,admitted] at success

#axiomsClean aggregate_success
#axiomsClean forward_launch_success
#axiomsClean backward_launch_success
#axiomsClean forward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Invocation
