import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Host
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.AggregateTrace

namespace VeriTile.Bench.Repairs.CrossEntropy2.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.Host (World Step bind LocalTrace)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000

structure ForwardData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ) (o : Options)
    (callerLogits callerLabels : Tensor) (rows cols : Nat) (initial : World α σ) (returned : Return) (final : World α σ) where
  admitted : valid o callerLogits callerLabels rows cols
  logits : Tensor
  labels : Tensor
  normalizedLogits : World α σ
  normalizedLabels : World α σ
  normalizeLogits : normalize rt callerLogits initial=.ok logits normalizedLogits
  normalizeLabels : normalize rt callerLabels normalizedLogits=.ok labels normalizedLabels
  loss : Tensor
  lse : Tensor
  zLoss : Tensor
  lossAllocated : World α σ
  lseAllocated : World α σ
  zAllocated : World α σ
  computed : World α σ
  count : Nat
  countEq : count=VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width
  split : Bool
  splitEq : split=(o.split || decide (count>1 ∨ o.start≠0 ∨ cols≠o.total))
  lossCall : rt.empty (if count>1 then [count,rows] else [rows]) "torch.float32" logits.device normalizedLabels=.ok loss lossAllocated
  lseCall : rt.empty (if count>1 then [count,rows] else [rows]) "torch.float32" logits.device lossAllocated=.ok lse lseAllocated
  zCall : rt.empty (if count>1 then [count,rows] else [rows]) "torch.float32" logits.device lseAllocated=.ok zLoss zAllocated
  launchCall : forwardLaunch M rt (parameters o logits labels loss lse zLoss rows cols split) count zAllocated=.ok () computed
  finishCall : (if split then aggregate rt o count labels loss lse computed else .ok (loss,lse,zLoss) computed)=.ok returned final

/-- Every recorded allocation and kernel setup is extracted from the actual
host result. This witness has no expected loss, LSE, gradient or final-byte field. -/
theorem forward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ) (o : Options)
    (callerLogits callerLabels : Tensor) (rows cols : Nat) (initial : World α σ) (returned : Return) (final : World α σ)
    (shape : callerLogits.shape=[rows,cols]) (success : forward M rt o callerLogits callerLabels initial=.ok returned final) :
    Nonempty (ForwardData M rt o callerLogits callerLabels rows cols initial returned final) := by
  classical
  by_cases admitted : valid o callerLogits callerLabels rows cols
  · simp only [forward,shape,if_pos admitted] at success
    obtain ⟨logits,sx,hx,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨labels,sy,hy,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨loss,sl,hl,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨lse,se,he,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨zLoss,sz,hz,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨u,computed,hlaunch,hfinish⟩ := (bind_success _ _ _ _).mp success
    cases u
    exact ⟨⟨admitted,logits,labels,sx,sy,hx,hy,loss,lse,zLoss,sl,se,sz,computed,_,rfl,_,rfl,hl,he,hz,hlaunch,hfinish⟩⟩
  · simp [forward,shape,admitted] at success

structure BackwardData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ) (o : Options)
    (callerUpstream callerLogits callerLSE callerLabels : Tensor) (rows cols : Nat)
    (initial : World α σ) (returned : Tensor) (final : World α σ) where
  admitted : valid o callerLogits callerLabels rows cols
  upstreamDtype : callerUpstream.dtype="torch.float32"
  lseDtype : callerLSE.dtype="torch.float32"
  upstreamShape : callerUpstream.shape=[rows]
  lseShape : callerLSE.shape=[rows]
  upstreamDevice : callerUpstream.device=callerLogits.device
  lseDevice : callerLSE.device=callerLogits.device
  logits : Tensor
  labels : Tensor
  upstream : Tensor
  lse : Tensor
  normalizedLogits : World α σ
  normalizedLabels : World α σ
  normalizedUpstream : World α σ
  normalizedLSE : World α σ
  allocated : World α σ
  normalizeLogits : normalize rt callerLogits initial=.ok logits normalizedLogits
  normalizeLabels : normalize rt callerLabels normalizedLogits=.ok labels normalizedLabels
  normalizeUpstream : normalize rt callerUpstream normalizedLabels=.ok upstream normalizedUpstream
  normalizeLSE : normalize rt callerLSE normalizedUpstream=.ok lse normalizedLSE
  allocationCall : rt.emptyLike logits Bool.true normalizedLSE=.ok returned allocated
  launchCall : backwardLaunch M rt
    {parameters o logits labels upstream lse lse rows cols Bool.false with
      gradient:=returned.region,gradientBase:=returned.storageOffset,
      gradientStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride returned,
      upstream:=upstream.region,upstreamBase:=upstream.storageOffset,
      upstreamStride:=VeriTile.Bench.Repairs.CrossEntropy1.Host.rowStride upstream}
    (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols o.width) allocated=.ok () final

theorem backward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ) (o : Options)
    (upstream logits lse labels : Tensor) (rows cols : Nat) (initial : World α σ) (returned : Tensor) (final : World α σ)
    (shape : logits.shape=[rows,cols]) (success : backward M rt o upstream logits lse labels initial=.ok returned final) :
    Nonempty (BackwardData M rt o upstream logits lse labels rows cols initial returned final) := by
  classical
  let admitted := valid o logits labels rows cols ∧
    upstream.dtype="torch.float32" ∧ lse.dtype="torch.float32" ∧ upstream.shape=[rows] ∧ lse.shape=[rows] ∧
    upstream.device=logits.device ∧ lse.device=logits.device
  by_cases h : admitted
  · dsimp only [admitted] at h
    simp only [backward,shape,if_pos h] at success
    obtain ⟨x,sx,hx,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨y,sy,hy,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨g,sg,hg,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨l,sl,hl,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨out,allocated,ha,success⟩ := (bind_success _ _ _ _).mp success
    obtain ⟨u,computed,hlaunch,hreturn⟩ := (bind_success _ _ _ _).mp success
    cases u
    cases hreturn
    exact ⟨⟨h.1,h.2.1,h.2.2.1,h.2.2.2.1,h.2.2.2.2.1,h.2.2.2.2.2.1,h.2.2.2.2.2.2,
      x,y,g,l,sx,sy,sg,sl,allocated,hx,hy,hg,hl,ha,hlaunch⟩⟩
  · dsimp only [admitted] at h
    simp [backward,shape,h] at success

#axiomsClean forward_success
#axiomsClean backward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Host
