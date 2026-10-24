import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorFrame
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorComplete
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeForward
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeBackward
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeGradient
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNativeTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelBackward (PID)
open KernelReal (parameters)
open KernelTensor (Role View)
open KernelPacket (Call)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset outputActive outputOffset)
noncomputable section
abbrev Word := BitVec 32

def observed (isa : ISA) (literal : ℝ→Word) (read : Nat→Word) (a : Call) (role : Role)
    (v : View a.shape role) (t : State KernelNative.Term) (i : Nat) : Word :=
  (VeriTile.Torch.FP32.logicalValue
    (fun addr=>⟨KernelTensor.project (KernelNative.S.lower (KernelNative.sharedOps (wordOps isa)) literal)
      read a role v t addr⟩) v.tensor v.laws i).bits

theorem readback (isa : ISA) (literal : ℝ→Word) (read : Nat→Word) (a : Call) (role : Role)
    (v : View a.shape role) (t : State KernelNative.Term) (off : Nat)
    (hb : off<v.tensor.metadata.numel) (hw : KernelPacket.Footprint a role.region off)
    (value : Word) (stored : KernelNative.observe isa literal t.mem role.region off=value) :
    observed isa literal read a role v t off=value := by
  rw [observed,KernelTensor.native_logical]
  exact (KernelTensor.projected (KernelNative.S.lower (KernelNative.sharedOps (wordOps isa)) literal)
    read a role v t off hb hw).trans stored

/-- Native Tensor dQ/dK/partial-dV equal the independent source-precision scalar
FMA references. Bounds follow from PID guards and the actual Tensor descriptors. -/
theorem gradient_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (scale : ℝ) (pid : PID) (m : RegionName→Nat→Word) (read : Nat→Word)
    (vq : View c .queryGradient) (vk : View c .keyGradient) (vv : View c .valuePartials)
    (hp : KernelTensor.G.GradientPID (parameters c) pid) :
    let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
    let ops:=wordOps (Numeric.referenceISA u)
    let xs:=KernelNative.data ops c m Bool.false Bool.false
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      KernelTensor.Frame (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
        read a (KernelNative.input m) t ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .queryGradient vq t (KernelGradient.gradKeyOffset (parameters c) pid i)=
          Body.dq ops c xs (literal scale) pid.z pid.y (KernelNative.history c pid m) i.1.val (pid.x*c.BK+i.2.1.val)) ∧
      (∀ i : TileIndex [64,c.BK],KernelGradient.gradKeyActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .keyGradient vk t (KernelGradient.gradKeyOffset (parameters c) pid i)=
          Body.dk ops c xs (literal scale) pid.z pid.y (KernelNative.dhistory c pid m) i.1.val (pid.x*c.BK+i.2.1.val)) ∧
      (∀ vb,vb<c.valueBlocks → ∀ i : TileIndex [64,c.BV],KernelGradient.gradValueActive (parameters c) pid vb i →
        observed (Numeric.nativeISA ports) literal read a .valuePartials vv t (KernelGradient.gradValueOffset (parameters c) pid (c.B*c.H) vb i)=
          Body.dvPartial ops c xs (literal scale) pid.x pid.z pid.y (KernelNative.dhistory c pid m) i.1.val (vb*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.backwardGradient,scale,Bool.false,Bool.false,pid⟩
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  obtain ⟨hq,hk,hv⟩:=KernelNative.gradient_target (Numeric.referenceISA u) literal constants c scale pid m hp.2.2
  refine ⟨t,ht,KernelTensor.frame_of_exec KernelNative.model (KernelNative.dotModel a)
    (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
    read a (KernelNative.input m) t ht,?_,?_,?_⟩
  · intro i hi
    exact readback _ literal read a .queryGradient vq t _
      (KernelTensor.gradient_bound vq (Or.inl rfl) pid hp i hi) (Or.inl ⟨Or.inl rfl,i,hi,rfl⟩) _
      ((hm "queryGradient" _).trans (hq i hi))
  · intro i hi
    exact readback _ literal read a .keyGradient vk t _
      (KernelTensor.gradient_bound vk (Or.inr rfl) pid hp i hi) (Or.inl ⟨Or.inr rfl,i,hi,rfl⟩) _
      ((hm "keyGradient" _).trans (hk i hi))
  · intro vb hvb i hi
    exact readback _ literal read a .valuePartials vv t _
      (KernelTensor.partial_bound vv pid hp vb i hi) (Or.inr ⟨rfl,vb,hvb,i,hi,rfl⟩) _
      ((hm "valuePartials" _).trans (hv vb hvb i hi))

theorem backward_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (scale : ℝ) (terminal initial : Bool) (pid : PID) (m : RegionName→Nat→Word) (read : Nat→Word)
    (vh : View c .dhistory) (vi : View c .initialGradient) (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
    let ops:=wordOps (Numeric.referenceISA u)
    let xs:=KernelNative.data ops c m Bool.false terminal
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      KernelTensor.Frame (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
        read a (KernelNative.input m) t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .dhistory vh t (hOffset (parameters c) pid (KernelBackward.time (parameters c) n) i)=
          ops.cast (Reference.reverseCarry ops c xs (literal scale) pid.z n (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (initial=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .initialGradient vi t (fOffset (parameters c) pid i)=
          ops.cast (Reference.reverseCarry ops c xs (literal scale) pid.z c.chunks (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) := by
  dsimp only
  let a : Call:=⟨c,.backwardState,scale,terminal,initial,pid⟩
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  obtain ⟨hh,hi⟩:=KernelNative.backward_target (Numeric.referenceISA u) literal constants c scale terminal initial pid m
  refine ⟨t,ht,KernelTensor.frame_of_exec KernelNative.model (KernelNative.dotModel a)
    (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
    read a (KernelNative.input m) t ht,?_,?_⟩
  · intro n hn i active
    exact readback _ literal read a .dhistory vh t _
      (KernelTensor.history_bound vh (Or.inr rfl) pid hp _
        (VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.chunk_lt (parameters c) Bool.true n hn) i active)
      (Or.inl ⟨rfl,n,hn,i,active,rfl⟩) _ ((hm "dhistory" _).trans (hh n hn i active))
  · intro flag i active
    exact readback _ literal read a .initialGradient vi t _
      (KernelTensor.state_bound vi (Or.inr rfl) pid hp i active)
      (Or.inr ⟨rfl,flag,i,active,rfl⟩) _ ((hm "initialGradient" _).trans (hi flag i active))

theorem output_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (scale : ℝ) (pid : PID) (m : RegionName→Nat→Word) (read : Nat→Word)
    (v : View c .output) (hp : KernelTensor.G.OutputPID (parameters c) pid) :
    let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
    let ops:=KernelNative.sharedOps (wordOps (Numeric.referenceISA u))
    let s:=KernelOutput.state c pid (KernelNative.input m)
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      KernelTensor.Frame (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
        read a (KernelNative.input m) t ∧
      (∀ i : TileIndex [64,c.BV],outputActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .output v t (outputOffset (parameters c) pid i)=
          VeriTile.Bench.Repairs.ChunkRetention.Whole001.Body.output ops (KernelOutput.shape c)
            (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.xs ops literal (KernelOutput.shape c) s)
            (literal scale) pid.z pid.y
            (VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.O.history ops literal (KernelOutput.shape c) s)
            i.1.val (pid.x*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardOutput,scale,Bool.false,Bool.false,pid⟩
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  refine ⟨t,ht,KernelTensor.frame_of_exec KernelNative.model (KernelNative.dotModel a)
    (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
    read a (KernelNative.input m) t ht,?_⟩
  intro i hi
  exact readback _ literal read a .output v t _ (KernelTensor.output_bound v pid hp i hi) ⟨rfl,i,hi,rfl⟩ _
    ((hm "o" _).trans (KernelNative.output_target (Numeric.referenceISA u) literal constants c scale pid m i hi))

theorem forward_tensor (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→Word)
    (constants : KernelNative.S.LiteralLaws (KernelNative.sharedOps (wordOps (Numeric.referenceISA u))) literal)
    (c : Shape) (seed final : Bool) (pid : PID) (m : RegionName→Nat→Word) (read : Nat→Word)
    (vh : View c .history) (vf : View c .finalState) (hp : KernelTensor.G.StatePID (parameters c) pid) :
    let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
    let ops:=KernelNative.sharedOps (wordOps (Numeric.referenceISA u))
    let xs:=VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelNative.H.xs ops literal (KernelOutput.shape c) seed
      (KernelForward.state c pid (KernelNative.input m))
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      KernelTensor.Frame (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
        read a (KernelNative.input m) t ∧
      (∀ n,n<c.chunks → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .history vh t (hOffset (parameters c) pid n i)=
          ops.cast (VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.history ops (KernelOutput.shape c) xs pid.z n
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val))) ∧
      (final=Bool.true → ∀ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i →
        observed (Numeric.nativeISA ports) literal read a .finalState vf t (fOffset (parameters c) pid i)=
          VeriTile.Bench.Repairs.ChunkRetention.Whole001.Reference.finalState ops (KernelOutput.shape c) xs pid.z
            (pid.x*c.BK+i.1.val) (pid.y*c.BV+i.2.1.val)) := by
  dsimp only
  let a : Call:=⟨c,.forwardState,0,seed,final,pid⟩
  obtain ⟨t,ht,hm,_⟩:=KernelNative.correct u ports laws literal a m
  obtain ⟨hh,hf⟩:=KernelNative.forward_target (Numeric.referenceISA u) literal constants c seed final pid m
  refine ⟨t,ht,KernelTensor.frame_of_exec KernelNative.model (KernelNative.dotModel a)
    (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
    read a (KernelNative.input m) t ht,?_,?_⟩
  · intro n hn i hi
    exact readback _ literal read a .history vh t _ (KernelTensor.history_bound vh (Or.inl rfl) pid hp n hn i hi)
      (Or.inl ⟨rfl,n,hn,i,hi,rfl⟩) _ ((hm "h" _).trans (hh n hn i hi))
  · intro flag i hi
    exact readback _ literal read a .finalState vf t _ (KernelTensor.state_bound vf (Or.inl rfl) pid hp i hi)
      (Or.inr ⟨rfl,flag,i,hi,rfl⟩) _ ((hm "ht" _).trans (hf flag i hi))

#axiomsClean gradient_tensor
#axiomsClean backward_tensor
#axiomsClean output_tensor
#axiomsClean forward_tensor
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelNativeTensor
