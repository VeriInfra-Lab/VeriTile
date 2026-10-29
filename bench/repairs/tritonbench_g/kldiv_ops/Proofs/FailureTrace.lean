import bench.repairs.tritonbench_g.kldiv_ops.Proofs.HostBackward

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage
noncomputable section

/-- An exact failed action, including its actual state. Successful prefixes
are recorded, so allocation, capacity, copy, and post-copy failures remain distinct. -/
inductive FailureAt : Action A→State→String→State→Prop where
  | here (action : Action A) (s failed : State) (e : String)
      (call : action s=.error e failed) : FailureAt action s e failed

 theorem bind_failure {first : Action A} {next : A→Action B} {s failed : State} {e : String}
    (h : bind first next s=.error e failed) :
    first s=.error e failed ∨ ∃ a middle,first s=.ok a middle ∧ next a middle=.error e failed := by
  cases call : first s with
  | error detail t=>left; simpa [bind,call] using h
  | ok a t=>right; exact ⟨a,t,rfl,by simpa [bind,call] using h⟩

/-- Failed capacity validation occurs after packing and allocation, before
launch. It preserves that precise intermediate state without rollback. -/
theorem forward_capacity_failure {Word : Type} (api : Services Word) (source : SourceActions Word)
    (prediction target out py pt : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry configured allocated packed ready : State) (rows cols : Nat) (red : Reduction)
    (shape : shape2 prediction entry=.ok (rows,cols) entry)
    (cfg : api.configure rows cols entry=.ok () configured)
    (lookupCall : lookup name configured=.ok red configured)
    (alloc : api.zeros (red.outputShape rows cols) prediction.device configured=.ok out allocated)
    (predCall : prepare api prediction allocated=.ok py packed)
    (targetCall : prepare api target packed=.ok pt ready) (short : pt.numel<rows*cols) :
    forward api source prediction target flag name eps entry=
      .error "target has fewer elements than prediction" ready := by
  simp [forward,Bind.bind,bind,shape,cfg,lookupCall,alloc,predCall,targetCall,
    capacity_failure pt (rows*cols) "target has fewer elements than prediction" ready short]

theorem backward_capacity_failure {Word : Type} (api : Services Word) (source : SourceActions Word)
    (target grad dst pt pg work : Tensor) (flag : Bool)
    (entry configured packed cloned ready : State) (rows cols : Nat)
    (shape : shape2 target entry=.ok (rows,cols) entry)
    (cfg : api.configure rows cols entry=.ok () configured)
    (targetCall : prepare api target configured=.ok pt packed)
    (gradCall : snapshot api grad packed=.ok pg cloned)
    (alloc : api.emptyContiguousLike dst cloned=.ok work ready) (short : work.numel<rows*cols) :
    backward api source target grad dst flag entry=
      .error "new_grads has fewer elements than target" ready := by
  simp [backward,Bind.bind,bind,shape,cfg,targetCall,gradCall,alloc,
    capacity_failure work (rows*cols) "new_grads has fewer elements than target" ready short]

/-- Every failure of the final backward path is a copy, scalar construction,
equality or multiplication failure, at the actual corresponding state. -/
theorem backward_finish_failure {Word : Type} (api : Services Word) (grad dst work : Tensor)
    (entry failed : State) (e : String)
    (h : backwardFinish api grad dst work entry=.error e failed) :
    api.copy dst work entry=.error e failed ∨
    (∃ copiedReturn copied,api.copy dst work entry=.ok copiedReturn copied ∧
      (api.scalarOne grad.device copied=.error e failed ∨
      ∃ one compared,api.scalarOne grad.device copied=.ok one compared ∧
        (api.equal grad one compared=.error e failed ∨
        ∃ tested,api.equal grad one compared=.ok Bool.false tested ∧
          api.multiply dst grad tested=.error e failed))) := by
  rcases bind_failure h with hc|⟨copiedReturn,copied,hc,h⟩
  · exact Or.inl hc
  · right
    refine ⟨copiedReturn,copied,hc,?_⟩
    rcases bind_failure h with ho|⟨one,compared,ho,h⟩
    · exact Or.inl ho
    · right
      refine ⟨one,compared,ho,?_⟩
      rcases bind_failure h with he|⟨test,tested,he,h⟩
      · exact Or.inl he
      · cases test with
        | «false»=>exact Or.inr ⟨tested,he,h⟩
        | «true»=>simp [pure] at h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
