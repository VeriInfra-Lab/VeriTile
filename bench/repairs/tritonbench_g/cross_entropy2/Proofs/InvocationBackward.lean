import bench.repairs.tritonbench_g.cross_entropy2.Proofs.InvocationTrace

namespace VeriTile.Bench.Repairs.CrossEntropy2.Invocation
open VeriTile.Triton Source Host Floating
open Floating.Profiled (DeviceFamily BackwardABI BackwardOrder BackwardSelected)
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World)
open VeriTile.Bench.Repairs.CrossEntropy1.Physical (Layout)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 10000000

theorem backward_success {σ : Type} (layout : Layout σ) (abi : BackwardABI) (device : DeviceFamily)
    (rt : Runtime Compiled σ) (o : Options) (g x l y : Tensor) (rows cols : Nat)
    (initial final : World Compiled σ) (order : BackwardOrder) (output : Tensor)
    (shape : x.shape=[rows,cols])
    (success : backward layout abi device rt o g x l y initial=.ok (order,output) final) :
    ∃ e : BackwardData (Floating.Profiled.algebra device o.logitScale) rt o g x l y rows cols initial output final,
      Host.backward (Floating.Profiled.algebra device o.logitScale) rt o g x l y initial=.ok output final ∧
      BackwardSelected layout abi order e := by
  classical
  let admitted := valid o x y rows cols ∧ g.dtype="torch.float32" ∧ l.dtype="torch.float32" ∧
    g.shape=[rows] ∧ l.shape=[rows] ∧ g.device=x.device ∧ l.device=x.device
  by_cases h : admitted
  · dsimp only [admitted] at h
    simp only [backward,shape,if_pos h] at success
    obtain ⟨logits,sx,hx,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨labels,sy,hy,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨upstream,sg,hg,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨lse,sl,hl,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨out,allocated,ha,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨chosen,computed,hlaunch,hreturn⟩ := (bind_success _ _ _ _).mp success
    cases hreturn
    have launch := backward_launch_success layout abi rt device _ _ _ _ _ hlaunch
    let e : BackwardData (Floating.Profiled.algebra device o.logitScale) rt o g x l y rows cols initial output final :=
      ⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2,
        logits,labels,upstream,lse,sx,sy,sg,sl,allocated,hx,hy,hg,hl,ha,launch.1⟩
    refine ⟨e,?_,launch.2⟩
    have originalLaunch := e.launchCall
    dsimp only [e] at originalLaunch
    simp only [Host.backward,shape,if_pos h,hx,hy,hg,hl,ha,originalLaunch,VeriTile.Bench.Repairs.CrossEntropy1.Host.bind]
  · dsimp only [admitted] at h
    simp [backward,shape,h] at success

#axiomsClean backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Invocation
