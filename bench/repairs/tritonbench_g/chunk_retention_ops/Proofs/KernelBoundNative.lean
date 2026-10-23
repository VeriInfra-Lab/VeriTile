import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorBacking
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeForward
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeBackward
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeGradient
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBoundNative
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters)
open KernelPacket (Call)
open KernelTensorBacking
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section
abbrev Word := BitVec 32

/-- All three gradient Tensors observe a single post-memory built from the
actual quoted state; all source operands read the same caller backing. -/
theorem gradient_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (scale : ℝ) (pid : PID) (read : Nat→Word)
    (f : Buffers ⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩)
    (hp : KernelTensor.G.GradientPID (parameters c) pid) :
    let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
    let m:=snapshot 0 read a f.inputs
    let ops:=wordOps (Numeric.referenceISA u)
    let xs:=KernelNative.data ops c m Bool.false Bool.false
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      Frame (nativeDecode (Numeric.nativeISA ports) literal) read a f t ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.queryGradient,by trivial⟩
          (KernelGradient.gradKeyOffset (parameters c) pid i)=
          Body.dq ops c xs (literal scale) pid.z pid.y (KernelNative.history c pid m)
            i.1.val (pid.x*c.BK+i.2.1.val)) ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.keyGradient,by trivial⟩
          (KernelGradient.gradKeyOffset (parameters c) pid i)=
          Body.dk ops c xs (literal scale) pid.z pid.y (KernelNative.dhistory c pid m)
            i.1.val (pid.x*c.BK+i.2.1.val)) ∧
      (∀ vb,vb<c.valueBlocks → ∀ i : TileIndex [64,c.BV],KernelGradient.gradValueActive (parameters c) pid vb i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.valuePartials,by trivial⟩
          (KernelGradient.gradValueOffset (parameters c) pid (c.B*c.H) vb i)=
          Body.dvPartial ops c xs (literal scale) pid.x pid.z pid.y (KernelNative.dhistory c pid m)
            i.1.val (vb*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
  let m:=snapshot 0 read a f.inputs
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  obtain ⟨hq,hk,hv⟩:=KernelNative.gradient_target (Numeric.referenceISA u) literal constants c scale pid m hp.2.2
  refine ⟨t,ht,frame _ read a f t,?_,?_,?_⟩
  · intro i hi
    exact native_readback _ literal read a f t hp ⟨.queryGradient,by trivial⟩ _
      (Or.inl ⟨Or.inl rfl,i,hi,rfl⟩) _ ((hm "queryGradient" _).trans (hq i hi))
  · intro i hi
    exact native_readback _ literal read a f t hp ⟨.keyGradient,by trivial⟩ _
      (Or.inl ⟨Or.inr rfl,i,hi,rfl⟩) _ ((hm "keyGradient" _).trans (hk i hi))
  · intro vb hvb i hi
    exact native_readback _ literal read a f t hp ⟨.valuePartials,by trivial⟩ _
      (Or.inr ⟨rfl,vb,hvb,i,hi,rfl⟩) _ ((hm "valuePartials" _).trans (hv vb hvb i hi))

theorem backward_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (scale : ℝ) (terminal initial : Bool) (pid : PID) (read : Nat→Word)
    (f : Buffers ⟨c,.backwardState,scale,terminal,initial,pid⟩)
    (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
    let m:=snapshot 0 read a f.inputs
    let ops:=wordOps (Numeric.referenceISA u)
    let xs:=KernelNative.data ops c m Bool.false terminal
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      Frame (nativeDecode (Numeric.nativeISA ports) literal) read a f t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.dhistory,by trivial⟩
          (hOffset (parameters c) pid (KernelBackward.time (parameters c) n) i)=
          ops.cast (Reference.reverseCarry ops c xs (literal scale) pid.z n
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (∀ flag : initial=Bool.true,∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.initialGradient,flag⟩
          (fOffset (parameters c) pid i)=
          ops.cast (Reference.reverseCarry ops c xs (literal scale) pid.z c.chunks
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) := by
  dsimp only
  let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
  let m:=snapshot 0 read a f.inputs
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  obtain ⟨hh,hi⟩:=KernelNative.backward_target (Numeric.referenceISA u) literal constants c scale terminal initial pid m
  refine ⟨t,ht,frame _ read a f t,?_,?_⟩
  · intro n hn i active
    exact native_readback _ literal read a f t hp ⟨.dhistory,by trivial⟩ _
      (Or.inl ⟨rfl,n,hn,i,active,rfl⟩) _ ((hm "dhistory" _).trans (hh n hn i active))
  · intro flag i active
    exact native_readback _ literal read a f t hp ⟨.initialGradient,flag⟩ _
      (Or.inr ⟨rfl,flag,i,active,rfl⟩) _ ((hm "initialGradient" _).trans (hi flag i active))

theorem output_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (scale : ℝ) (pid : PID) (read : Nat→Word)
    (f : Buffers ⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩)
    (hp : KernelTensor.G.OutputPID (parameters c) pid) :
    let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
    let m:=snapshot 0 read a f.inputs
    let ops:=KernelNative.sharedOps (wordOps (Numeric.referenceISA u))
    let s:=KernelOutput.state c pid (KernelNative.input m)
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      Frame (nativeDecode (Numeric.nativeISA ports) literal) read a f t ∧
      (∀ i : TileIndex [64,c.BV],outputActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.output,by trivial⟩
          (outputOffset (parameters c) pid i)=
          VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.output ops (KernelOutput.shape c)
            (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.xs ops literal (KernelOutput.shape c) s)
            (literal scale) pid.z pid.y
            (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.history ops literal (KernelOutput.shape c) s)
            i.1.val (pid.x*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
  let m:=snapshot 0 read a f.inputs
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  refine ⟨t,ht,frame _ read a f t,?_⟩
  intro i hi
  exact native_readback _ literal read a f t hp ⟨.output,by trivial⟩ _ ⟨rfl,i,hi,rfl⟩ _
    ((hm "o" _).trans (KernelNative.output_target (Numeric.referenceISA u) literal constants c scale pid m i hi))

theorem forward_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (seed final : Bool) (pid : PID) (read : Nat→Word)
    (f : Buffers ⟨c,.forwardState,0,seed,final,pid⟩)
    (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
    let m:=snapshot 0 read a f.inputs
    let ops:=KernelNative.sharedOps (wordOps (Numeric.referenceISA u))
    let xs:=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.H.xs ops literal (KernelOutput.shape c) seed
      (KernelForward.state c pid (KernelNative.input m))
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      Frame (nativeDecode (Numeric.nativeISA ports) literal) read a f t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.history,by trivial⟩
          (hOffset (parameters c) pid n i)=
          ops.cast (VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history ops (KernelOutput.shape c) xs pid.z n
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (∀ flag : final=Bool.true,∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        nativeObserved (Numeric.nativeISA ports) literal read a f t ⟨.finalState,flag⟩
          (fOffset (parameters c) pid i)=
          VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.finalState ops (KernelOutput.shape c) xs pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
  let m:=snapshot 0 read a f.inputs
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  obtain ⟨hh,hf⟩:=KernelNative.forward_target (Numeric.referenceISA u) literal constants c seed final pid m
  refine ⟨t,ht,frame _ read a f t,?_,?_⟩
  · intro n hn i hi
    exact native_readback _ literal read a f t hp ⟨.history,by trivial⟩ _
      (Or.inl ⟨rfl,n,hn,i,hi,rfl⟩) _ ((hm "h" _).trans (hh n hn i hi))
  · intro flag i hi
    exact native_readback _ literal read a f t hp ⟨.finalState,flag⟩ _
      (Or.inr ⟨rfl,flag,i,hi,rfl⟩) _ ((hm "ht" _).trans (hf flag i hi))

#axiomsClean gradient_tensor
#axiomsClean backward_tensor
#axiomsClean output_tensor
#axiomsClean forward_tensor
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBoundNative
