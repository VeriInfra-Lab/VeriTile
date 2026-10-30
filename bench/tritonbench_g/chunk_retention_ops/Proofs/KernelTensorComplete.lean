import bench.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorBounds
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open KernelPacket (Call)
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (withDot)
noncomputable section
set_option maxHeartbeats 6000000

def Valid (a : Call) : Prop := match a.stage with
  | .forwardState | .backwardState=>G.StatePID (parameters a.shape) a.pid
  | .forwardOutput=>G.OutputPID (parameters a.shape) a.pid
  | .backwardGradient=>G.GradientPID (parameters a.shape) a.pid

/-- All observation bounds follow from the existing kernel PID guards and the
actual Tensor shape; none is an additional numerical/domain assumption. -/
theorem footprint_bound (a : Call) (hp : Valid a) (role : Role) (v : View a.shape role)
    (off : Nat) (hw : KernelPacket.Footprint a role.region off) : off<v.tensor.metadata.numel := by
  cases hstage : a.stage with
  | forwardState=>
    simp only [Valid,hstage] at hp
    cases role <;> simp [KernelPacket.Footprint,hstage,KernelForward.Footprint,Role.region] at hw
    · rcases hw with ⟨n,hn,x,y,z,hi,rfl⟩
      exact history_bound v (Or.inl rfl) a.pid hp n hn ⟨x,y,z⟩ hi
    · rcases hw with ⟨_,x,y,z,hi,rfl⟩
      exact state_bound v (Or.inl rfl) a.pid hp ⟨x,y,z⟩ hi
  | forwardOutput=>
    simp only [Valid,hstage] at hp
    cases role <;> simp [KernelPacket.Footprint,hstage,KernelOutput.Footprint,Role.region] at hw
    rcases hw with ⟨x,y,z,hi,rfl⟩
    exact output_bound v a.pid hp ⟨x,y,z⟩ hi
  | backwardState=>
    simp only [Valid,hstage] at hp
    cases role <;> simp [KernelPacket.Footprint,hstage,KernelPacket.backwardFootprint,Role.region] at hw
    · rcases hw with ⟨n,hn,x,y,z,hi,rfl⟩
      exact history_bound v (Or.inr rfl) a.pid hp _
        (VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry.chunk_lt (parameters a.shape) Bool.true n hn) ⟨x,y,z⟩ hi
    · rcases hw with ⟨_,x,y,z,hi,rfl⟩
      exact state_bound v (Or.inr rfl) a.pid hp ⟨x,y,z⟩ hi
  | backwardGradient=>
    simp only [Valid,hstage] at hp
    cases role <;> simp [KernelPacket.Footprint,hstage,VeriTile.Bench.Repairs.ChunkLinearAttn.GradientFootprint,Role.region] at hw
    · rcases hw with ⟨x,y,z,hi,rfl⟩
      exact gradient_bound v (Or.inl rfl) a.pid hp ⟨x,y,z⟩ hi
    · rcases hw with ⟨x,y,z,hi,rfl⟩
      exact gradient_bound v (Or.inr rfl) a.pid hp ⟨x,y,z⟩ hi
    · rcases hw with ⟨vb,_,x,y,z,hi,rfl⟩
      exact partial_bound v a.pid hp vb ⟨x,y,z⟩ hi

/-- Native public endpoint for every source-written logical Tensor element.
The only execution guards are the existing kernel PID conditions. -/
theorem native_complete (u : Numeric.UnaryModels) (ports : Numeric.ScalarPorts)
    (laws : Numeric.ScalarLaws u ports) (literal : ℝ→BitVec 32) (read : Nat→BitVec 32)
    (a : Call) (hp : Valid a) (role : Role) (v : View a.shape role) (m : RegionName→Nat→BitVec 32) :
    ∃ t,FP.Structural.exec (engine (withDot KernelNative.model (KernelNative.dotModel a)))
        a.kernel (a.state (KernelNative.input m))=some t ∧
      (∀ i,KernelPacket.Footprint a role.region i →
        (VeriTile.Torch.FP32.logicalValue
          (fun addr=>⟨project (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
            read a role v t addr⟩) v.tensor v.laws i).bits=
          KernelNative.target (Numeric.referenceISA u) literal a m role.region i) ∧
      (∀ addr,¬Written a role v addr →
        project (KernelNative.S.lower (KernelNative.sharedOps (wordOps (Numeric.nativeISA ports))) literal)
          read a role v t addr=read addr) := by
  obtain ⟨t,ht,hv,hf⟩:=native_tensor u ports laws literal read a role v m
  exact ⟨t,ht,fun i hi=>hv i (footprint_bound a hp role v i hi) hi,hf⟩

#axiomsClean footprint_bound
#axiomsClean native_complete
#specNonCircular Valid avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensor
