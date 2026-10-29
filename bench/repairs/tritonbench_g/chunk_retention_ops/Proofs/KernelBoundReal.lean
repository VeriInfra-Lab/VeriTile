import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorBacking
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientCorrect
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBoundReal
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters algebra dotPrimitive inputs)
open KernelPacket (Call)
open KernelTensorBacking
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section

/-- The exact-Real forward kernel reads genuine input Tensor backing and writes
history plus optional final state into one common separated output backing. -/
theorem forward_tensor (c : Shape) (seed final : Bool) (pid : PID) (read : Nat→ℝ)
    (f : Buffers ⟨c,.forwardState,0,seed,final,pid⟩)
    (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
    let m:=realInput read a f.inputs
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      Frame id read a f t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        realObserved read a f t ⟨.history,by trivial⟩ (hOffset (parameters c) pid n i)=
          MathTarget.state c (inputs c m seed Bool.false) Bool.true pid.z (c.boundary n)
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) ∧
      (∀ flag : final=Bool.true,∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        realObserved read a f t ⟨.finalState,flag⟩ (fOffset (parameters c) pid i)=
          MathTarget.finalState c (inputs c m seed Bool.false) Bool.true pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
  let m:=realInput read a f.inputs
  obtain ⟨t,ht,hh,hf,_⟩:=KernelReal.forward_correct c seed final pid m
  refine ⟨t,ht,frame id read a f t,?_,?_⟩
  · intro n hn i hi
    exact real_readback read a f t hp ⟨.history,by trivial⟩ _
      (Or.inl ⟨rfl,n,hn,i,hi,rfl⟩) _ (hh n hn i hi)
  · intro flag i hi
    exact real_readback read a f t hp ⟨.finalState,flag⟩ _
      (Or.inr ⟨rfl,flag,i,hi,rfl⟩) _ (hf flag i hi)

theorem output_tensor (c : Shape) (scale : ℝ) (pid : PID) (read : Nat→ℝ)
    (f : Buffers ⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩)
    (hp : KernelTensor.G.OutputPID (parameters c) pid) :
    let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
    let m:=realInput read a f.inputs
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      Frame id read a f t ∧
      (∀ i : TileIndex [64,c.BV],outputActive (parameters c) pid i →
        realObserved read a f t ⟨.output,by trivial⟩ (outputOffset (parameters c) pid i)=
          ScalarSource.outputTile c (inputs c m Bool.false Bool.false) scale pid.z pid.y i.1.val
            (pid.x*c.BV+i.2.1.val) (KernelReal.O.history c (KernelOutput.state c pid m))) := by
  dsimp only
  let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
  let m:=realInput read a f.inputs
  obtain ⟨t,ht,hv,_⟩:=KernelReal.output_correct c scale pid m
  refine ⟨t,ht,frame id read a f t,?_⟩
  intro i hi
  exact real_readback read a f t hp ⟨.output,by trivial⟩ _ ⟨rfl,i,hi,rfl⟩ _ (hv i hi)

theorem backward_tensor (c : Shape) (scale : ℝ) (terminal initial : Bool) (pid : PID)
    (read : Nat→ℝ) (f : Buffers ⟨c,.backwardState,scale,terminal,initial,pid⟩)
    (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
    let m:=realInput read a f.inputs
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      Frame id read a f t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        realObserved read a f t ⟨.dhistory,by trivial⟩
          (hOffset (parameters c) pid (KernelBackward.time (parameters c) n) i)=
          MathTarget.future c (inputs c m Bool.false terminal) scale pid.z (c.boundary (c.chunks-n))
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) ∧
      (∀ flag : initial=Bool.true,∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        realObserved read a f t ⟨.initialGradient,flag⟩ (fOffset (parameters c) pid i)=
          MathTarget.initialGradient c (inputs c m Bool.false terminal) scale pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
  let m:=realInput read a f.inputs
  obtain ⟨t,ht,hh,hi,_⟩:=KernelReal.backward_correct c scale Bool.false terminal initial (a.state m) hp
  refine ⟨t,ht,frame id read a f t,?_,?_⟩
  · intro n hn i active
    exact real_readback read a f t hp ⟨.dhistory,by trivial⟩ _
      (Or.inl ⟨rfl,n,hn,i,active,rfl⟩) _ (hh n hn i active)
  · intro flag i active
    exact real_readback read a f t hp ⟨.initialGradient,flag⟩ _
      (Or.inr ⟨rfl,flag,i,active,rfl⟩) _ (hi flag i active)

/-- dQ, dK and per-key-block partial dV are observations of the very same
post-memory, with every input bound to the same initial physical read. -/
theorem gradient_tensor (c : Shape) (scale : ℝ) (pid : PID) (read : Nat→ℝ)
    (f : Buffers ⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩)
    (hp : KernelTensor.G.GradientPID (parameters c) pid) :
    let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
    let m:=realInput read a f.inputs
    let xs:=inputs c m Bool.false Bool.false
    ∃ t,FP.Structural.exec (engine (withDot algebra dotPrimitive)) a.kernel (a.state m)=some t ∧
      Frame id read a f t ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        realObserved read a f t ⟨.queryGradient,by trivial⟩ (KernelGradient.gradKeyOffset (parameters c) pid i)=
          ScalarSource.queryTile c xs scale pid.z pid.y i.1.val (pid.x*c.BK+i.2.1.val) (KernelReal.history c pid m)) ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        realObserved read a f t ⟨.keyGradient,by trivial⟩ (KernelGradient.gradKeyOffset (parameters c) pid i)=
          ScalarSource.keyTile c xs scale pid.z pid.y i.1.val (pid.x*c.BK+i.2.1.val) (KernelReal.dhistory c pid m)) ∧
      (∀ vb,vb<c.valueBlocks → ∀ i : TileIndex [64,c.BV],KernelGradient.gradValueActive (parameters c) pid vb i →
        realObserved read a f t ⟨.valuePartials,by trivial⟩
          (KernelGradient.gradValueOffset (parameters c) pid (c.B*c.H) vb i)=
          ScalarSource.valuePartial c xs scale pid.x pid.z pid.y i.1.val (vb*c.BV+i.2.1.val)
            (KernelReal.dhistory c pid m)) := by
  dsimp only
  let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
  let m:=realInput read a f.inputs
  obtain ⟨t,ht,hq,hk,hv,_⟩:=KernelReal.gradient_correct c scale (a.state m) rfl hp.2.2 hp.2.1
  refine ⟨t,ht,frame id read a f t,?_,?_,?_⟩
  · intro i hi
    exact real_readback read a f t hp ⟨.queryGradient,by trivial⟩ _
      (Or.inl ⟨Or.inl rfl,i,hi,rfl⟩) _ (hq i hi)
  · intro i hi
    exact real_readback read a f t hp ⟨.keyGradient,by trivial⟩ _
      (Or.inl ⟨Or.inr rfl,i,hi,rfl⟩) _ (hk i hi)
  · intro vb hvb i hi
    exact real_readback read a f t hp ⟨.valuePartials,by trivial⟩ _
      (Or.inr ⟨rfl,vb,hvb,i,hi,rfl⟩) _ (hv vb hvb i hi)

#axiomsClean forward_tensor
#axiomsClean output_tensor
#axiomsClean backward_tensor
#axiomsClean gradient_tensor
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBoundReal
