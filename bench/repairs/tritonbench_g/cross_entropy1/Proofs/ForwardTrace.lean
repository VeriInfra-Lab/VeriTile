import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Library

namespace VeriTile.Bench.Repairs.CrossEntropy1.Host
open VeriTile.Triton Source
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

structure ForwardData {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (callerLogits callerLabels : Tensor) (initial : World α σ)
    (returned : ForwardReturn) (final : World α σ) where
  rows : Nat
  cols : Nat
  logitsShape : callerLogits.shape=[rows,cols]
  labelsShape : callerLabels.shape=[rows]
  world : Nat
  sized : World α σ
  sizeCall : groupSize rt o.processGroup initial=Step.ok world sized
  rank : Nat
  ranked : World α σ
  rankCall : groupRank rt o.processGroup sized=Step.ok rank ranked
  labels : Tensor
  labelsReady : World α σ
  snapshotCall : snapshot rt callerLabels ranked=.ok labels labelsReady
  inplaceLogits : Tensor
  normalized : World α σ
  originalNormalize : (if lastStride callerLogits≠1 then rt.contiguous callerLogits labelsReady else
    .ok callerLogits labelsReady)=Step.ok inplaceLogits normalized
  overlapGuard : ¬(o.inplace && decide (rows>1 ∧ rowStride inplaceLogits<cols))=Bool.true
  resolvedLogits : Tensor
  resolvedState : World α σ
  resolveCall : rt.resolveNeg inplaceLogits normalized=.ok resolvedLogits resolvedState
  workingLogits : Tensor
  resolved : World α σ
  workingNormalize : (if lastStride resolvedLogits≠1 then rt.contiguous resolvedLogits resolvedState else
    .ok resolvedLogits resolvedState)=Step.ok workingLogits resolved
  nonzero : fwdWidth cols≠0
  losses : Tensor
  lossAllocated : World α σ
  lossCall : rt.empty (if Math.blocks cols (fwdWidth cols)>1 then [Math.blocks cols (fwdWidth cols),rows] else [rows])
    "torch.float32" workingLogits.device resolved=.ok losses lossAllocated
  lse : Tensor
  lseAllocated : World α σ
  lseCall : rt.empty (if Math.blocks cols (fwdWidth cols)>1 then [Math.blocks cols (fwdWidth cols),rows] else [rows])
    "torch.float32" workingLogits.device lossAllocated=.ok lse lseAllocated
  computed : World α σ
  launchCall : forwardLaunch M rt workingLogits.device
    (forwardParameters o workingLogits labels losses lse rows cols (world*cols) (rank*cols) (fwdWidth cols)
      (decide (world>1 ∨ cols>65536))) (Math.blocks cols (fwdWidth cols)) lseAllocated=.ok () computed
  pair : Tensor × Tensor
  aggregated : World α σ
  aggregateCall : (if decide (world>1 ∨ cols>65536) then
    aggregate rt o world rows (Math.blocks cols (fwdWidth cols)) labels losses lse computed else .ok (losses,lse) computed)=
      Step.ok pair aggregated
  saveCall : rt.saveTensors workingLogits pair.2 labels inplaceLogits aggregated=.ok () final
  returnIdentity : returned=⟨pair.1,⟨workingLogits,pair.2,labels,inplaceLogits,o,world*cols,rank*cols⟩⟩

theorem forward_success {α σ : Type} [Inhabited α] (M : FP32 α) (rt : Runtime α σ)
    (o : Options) (callerLogits callerLabels : Tensor) (initial : World α σ)
    (returned : ForwardReturn) (final : World α σ)
    (success : forward M rt o callerLogits callerLabels initial=.ok returned final) :
    Nonempty (ForwardData M rt o callerLogits callerLabels initial returned final) := by
  unfold forward at success
  split at success
  next rows cols logitsShape =>
    by_cases labelsShape : callerLabels.shape=[rows]
    · rw [if_pos labelsShape] at success
      cases hsize : groupSize rt o.processGroup initial with
      | error reason failed => simp [bind,hsize] at success
      | ok world sized =>
        simp only [hsize,bind] at success
        cases hrank : groupRank rt o.processGroup sized with
        | error reason failed => simp [bind,hrank] at success
        | ok rank ranked =>
          simp only [hrank,bind] at success
          cases hlabels : snapshot rt callerLabels ranked with
          | error reason failed => simp [bind,hlabels] at success
          | ok labels labelsReady =>
            simp only [hlabels,bind] at success
            cases hnormalize : (if lastStride callerLogits≠1 then rt.contiguous callerLogits labelsReady
                else .ok callerLogits labelsReady) with
            | error reason failed => simp [bind,hnormalize] at success
            | ok inplaceLogits normalized =>
              simp only [hnormalize,bind] at success
              by_cases overlap : (o.inplace && decide (rows>1 ∧ rowStride inplaceLogits<cols))=Bool.true
              · rw [if_pos overlap] at success
                cases success
              · rw [if_neg overlap] at success
                cases hresolve : rt.resolveNeg inplaceLogits normalized with
                | error reason failed => simp [bind,hresolve] at success
                | ok resolvedLogits resolvedState =>
                  simp only [hresolve,bind] at success
                  cases hworking : (if lastStride resolvedLogits≠1 then rt.contiguous resolvedLogits resolvedState
                      else .ok resolvedLogits resolvedState) with
                  | error reason failed => simp [bind,hworking] at success
                  | ok workingLogits resolved =>
                    simp only [hworking,bind] at success
                    by_cases zero : fwdWidth cols=0
                    · rw [if_pos zero] at success
                      cases success
                    · rw [if_neg zero] at success
                      cases hloss : rt.empty
                          (if Math.blocks cols (fwdWidth cols)>1 then [Math.blocks cols (fwdWidth cols),rows] else [rows])
                          "torch.float32" workingLogits.device resolved with
                      | error reason failed => simp [bind,hloss] at success
                      | ok losses lossAllocated =>
                        simp only [hloss,bind] at success
                        cases hlse : rt.empty
                            (if Math.blocks cols (fwdWidth cols)>1 then [Math.blocks cols (fwdWidth cols),rows] else [rows])
                            "torch.float32" workingLogits.device lossAllocated with
                        | error reason failed => simp [bind,hlse] at success
                        | ok lse lseAllocated =>
                          simp only [hlse,bind] at success
                          cases hlaunch : forwardLaunch M rt workingLogits.device
                              (forwardParameters o workingLogits labels losses lse rows cols (world*cols) (rank*cols) (fwdWidth cols)
                                (decide (world>1 ∨ cols>65536))) (Math.blocks cols (fwdWidth cols)) lseAllocated with
                          | error reason failed =>
                            simp only [hlaunch,bind] at success
                            cases success
                          | ok u computed =>
                            cases u
                            simp only [hlaunch,bind] at success
                            cases haggregate : (if decide (world>1 ∨ cols>65536) then
                                aggregate rt o world rows (Math.blocks cols (fwdWidth cols)) labels losses lse computed
                                else .ok (losses,lse) computed) with
                            | error reason failed =>
                              simp only [haggregate,bind] at success
                              cases success
                            | ok pair aggregated =>
                              simp only [haggregate,bind] at success
                              cases hsave : rt.saveTensors workingLogits pair.2 labels inplaceLogits aggregated with
                              | error reason failed => simp [bind,hsave] at success
                              | ok u saved =>
                                cases u
                                simp only [hsave,bind,Step.ok.injEq] at success
                                obtain ⟨rfl,rfl⟩ := success
                                exact ⟨⟨rows,cols,logitsShape,labelsShape,world,sized,hsize,rank,ranked,hrank,
                                  labels,labelsReady,hlabels,inplaceLogits,normalized,hnormalize,overlap,
                                  resolvedLogits,resolvedState,hresolve,workingLogits,resolved,hworking,zero,
                                  losses,lossAllocated,hloss,lse,lseAllocated,hlse,computed,hlaunch,
                                  pair,aggregated,haggregate,hsave,rfl⟩⟩
    · rw [if_neg labelsShape] at success
      cases success
  next other shape => cases success

#axiomsClean forward_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Host
