import bench.repairs.tritonbench_g.chunk_retention_ops.KernelOnly.Misc.NativeCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Misc.RawCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Misc.RawNative
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBoundReal
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBoundNative

/-! Complete retained Tensor applications. Real and native namespaces preserve
all four source stages without placing their Tensor interfaces in main files. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.TensorBridge

namespace Real
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters algebra dotPrimitive inputs)
open KernelPacket (Call)
open KernelTensorBacking
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section

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
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) :=
  KernelBoundReal.forward_tensor c seed final pid read f hp

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
            (pid.x*c.BV+i.2.1.val) (KernelReal.O.history c (KernelOutput.state c pid m))) :=
  KernelBoundReal.output_tensor c scale pid read f hp

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
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) :=
  KernelBoundReal.backward_tensor c scale terminal initial pid read f hp

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
            (KernelReal.dhistory c pid m)) :=
  KernelBoundReal.gradient_tensor c scale pid read f hp

#axiomsClean forward_tensor
#axiomsClean output_tensor
#axiomsClean backward_tensor
#axiomsClean gradient_tensor
end
end Real

namespace Native
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters)
open KernelPacket (Call)
open KernelTensorBacking
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section
abbrev Word := BitVec 32

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
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) :=
  KernelBoundNative.forward_tensor u ports laws literal constants c seed final pid read f hp

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
            i.1.val (pid.x*c.BV+i.2.1.val)) :=
  KernelBoundNative.output_tensor u ports laws literal constants c scale pid read f hp

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
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) :=
  KernelBoundNative.backward_tensor u ports laws literal constants c scale terminal initial pid read f hp

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
            i.1.val (vb*c.BV+i.2.1.val)) :=
  KernelBoundNative.gradient_tensor u ports laws literal constants c scale pid read f hp

#axiomsClean forward_tensor
#axiomsClean output_tensor
#axiomsClean backward_tensor
#axiomsClean gradient_tensor
end
end Native
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.TensorBridge
