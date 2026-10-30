import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorFrame
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorBounds
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientCorrect
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelRealTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters algebra dotPrimitive inputs)
open KernelTensor (Role View logical project)
open KernelPacket (Call)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section

def observed (read : Nat→ℝ) (a : Call) (role : Role) (v : View a.shape role) (t : State ℝ) (off : Nat) : ℝ :=
  logical (KernelTensor.project id read a role v t) v off

theorem readback (read : Nat→ℝ) (a : Call) (role : Role) (v : View a.shape role) (t : State ℝ)
    (off : Nat) (bound : off<v.tensor.metadata.numel) (written : KernelPacket.Footprint a role.region off)
    (value : ℝ) (stored : t.mem role.region off=.mk .real value) : observed read a role v t off=value := by
  unfold observed
  rw [KernelTensor.projected id read a role v t off bound written,stored]
  rfl

theorem forward_tensor (c : Shape) (seed final : Bool) (pid : PID) (m : KernelReal.Memory ℝ)
    (read : Nat→ℝ) (vh : View c .history) (vf : View c .finalState)
    (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      KernelTensor.Frame id read a m t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed read a .history vh t (hOffset (parameters c) pid n i)=
          MathTarget.state c (inputs c m seed Bool.false) Bool.true pid.z (c.boundary n)
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) ∧
      (final=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed read a .finalState vf t (fOffset (parameters c) pid i)=
          MathTarget.finalState c (inputs c m seed Bool.false) Bool.true pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
  obtain ⟨t,ht,hh,hf,_⟩:=KernelReal.forward_correct c seed final pid m
  refine ⟨t,ht,KernelTensor.frame_of_exec algebra dotPrimitive id read a m t ht,?_,?_⟩
  · intro n hn i hi
    have hb:=KernelTensor.history_bound vh (Or.inl rfl) pid hp n hn i hi
    exact readback read a .history vh t _ hb (Or.inl ⟨rfl,n,hn,i,hi,rfl⟩) _ (hh n hn i hi)
  · intro flag i hi
    have hb:=KernelTensor.state_bound vf (Or.inl rfl) pid hp i hi
    exact readback read a .finalState vf t _ hb (Or.inr ⟨rfl,flag,i,hi,rfl⟩) _ (hf flag i hi)

theorem output_tensor (c : Shape) (scale : ℝ) (pid : PID) (m : KernelReal.Memory ℝ)
    (read : Nat→ℝ) (v : View c .output) (hp : KernelTensor.G.OutputPID (parameters c) pid) :
    let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      KernelTensor.Frame id read a m t ∧
      (∀ i : TileIndex [64,c.BV],outputActive (parameters c) pid i →
        observed read a .output v t (outputOffset (parameters c) pid i)=
          ScalarSource.outputTile c (inputs c m Bool.false Bool.false) scale pid.z pid.y i.1.val
            (pid.x*c.BV+i.2.1.val) (KernelReal.O.history c (KernelOutput.state c pid m))) := by
  dsimp only
  let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
  obtain ⟨t,ht,hv,_⟩:=KernelReal.output_correct c scale pid m
  refine ⟨t,ht,KernelTensor.frame_of_exec algebra dotPrimitive id read a m t ht,?_⟩
  intro i hi
  have hb:=KernelTensor.output_bound v pid hp i hi
  exact readback read a .output v t _ hb ⟨rfl,i,hi,rfl⟩ _ (hv i hi)

theorem backward_tensor (c : Shape) (scale : ℝ) (terminal initial : Bool) (pid : PID)
    (m : KernelReal.Memory ℝ) (read : Nat→ℝ) (vh : View c .dhistory) (vi : View c .initialGradient)
    (hp : VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      KernelTensor.Frame id read a m t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed read a .dhistory vh t (hOffset (parameters c) pid (KernelBackward.time (parameters c) n) i)=
          MathTarget.future c (inputs c m Bool.false terminal) scale pid.z (c.boundary (c.chunks-n))
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) ∧
      (initial=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed read a .initialGradient vi t (fOffset (parameters c) pid i)=
          MathTarget.initialGradient c (inputs c m Bool.false terminal) scale pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
  obtain ⟨t,ht,hh,hi,_⟩:=KernelReal.backward_correct c scale Bool.false terminal initial (a.state m) hp
  refine ⟨t,ht,KernelTensor.frame_of_exec algebra dotPrimitive id read a m t ht,?_,?_⟩
  · intro n hn i active
    have htime : KernelBackward.time (parameters c) n<c.chunks:=VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.chunk_lt (parameters c) Bool.true n hn
    have hb:=KernelTensor.history_bound vh (Or.inr rfl) pid hp _ htime i active
    exact readback read a .dhistory vh t _ hb (Or.inl ⟨rfl,n,hn,i,active,rfl⟩) _ (hh n hn i active)
  · intro flag i active
    have hb:=KernelTensor.state_bound vi (Or.inr rfl) pid hp i active
    exact readback read a .initialGradient vi t _ hb (Or.inr ⟨rfl,flag,i,active,rfl⟩) _ (hi flag i active)

theorem gradient_tensor (c : Shape) (scale : ℝ) (pid : PID) (m : KernelReal.Memory ℝ)
    (read : Nat→ℝ) (vq : View c .queryGradient) (vk : View c .keyGradient) (vv : View c .valuePartials)
    (hp : KernelTensor.G.GradientPID (parameters c) pid) :
    let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
    let xs:=inputs c m Bool.false Bool.false
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      KernelTensor.Frame id read a m t ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        observed read a .queryGradient vq t (KernelGradient.gradKeyOffset (parameters c) pid i)=
          ScalarSource.queryTile c xs scale pid.z pid.y i.1.val (pid.x*c.BK+i.2.1.val) (KernelReal.history c pid m)) ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        observed read a .keyGradient vk t (KernelGradient.gradKeyOffset (parameters c) pid i)=
          ScalarSource.keyTile c xs scale pid.z pid.y i.1.val (pid.x*c.BK+i.2.1.val) (KernelReal.dhistory c pid m)) ∧
      (∀ vb,vb<c.valueBlocks → ∀ i : TileIndex [64,c.BV],KernelGradient.gradValueActive (parameters c) pid vb i →
        observed read a .valuePartials vv t (KernelGradient.gradValueOffset (parameters c) pid (c.B*c.H) vb i)=
          ScalarSource.valuePartial c xs scale pid.x pid.z pid.y i.1.val (vb*c.BV+i.2.1.val) (KernelReal.dhistory c pid m)) := by
  dsimp only
  let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
  obtain ⟨t,ht,hq,hk,hv,_⟩:=KernelReal.gradient_correct c scale (a.state m) rfl hp.2.2 hp.2.1
  refine ⟨t,ht,KernelTensor.frame_of_exec algebra dotPrimitive id read a m t ht,?_,?_,?_⟩
  · intro i hi
    have hb:=KernelTensor.gradient_bound vq (Or.inl rfl) pid hp i hi
    exact readback read a .queryGradient vq t _ hb (Or.inl ⟨Or.inl rfl,i,hi,rfl⟩) _ (hq i hi)
  · intro i hi
    have hb:=KernelTensor.gradient_bound vk (Or.inr rfl) pid hp i hi
    exact readback read a .keyGradient vk t _ hb (Or.inl ⟨Or.inr rfl,i,hi,rfl⟩) _ (hk i hi)
  · intro vb hvb i hi
    have hb:=KernelTensor.partial_bound vv pid hp vb i hi
    exact readback read a .valuePartials vv t _ hb (Or.inr ⟨rfl,vb,hvb,i,hi,rfl⟩) _ (hv vb hvb i hi)

#axiomsClean forward_tensor
#axiomsClean output_tensor
#axiomsClean backward_tensor
#axiomsClean gradient_tensor
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelRealTensor
