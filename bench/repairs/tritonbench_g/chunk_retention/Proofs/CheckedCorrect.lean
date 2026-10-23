import bench.repairs.tritonbench_g.chunk_retention.Proofs.RealReturn

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open scoped Classical
set_option maxHeartbeats 3500000

structure ForwardRealPost (c : Shape) (f : ForwardObjects c) (scale : ℝ) (tree : SumTree)
    (priorDevice : Nat) (before : RealMemory) (callers : List Tensor) (result : RealForwardReturn) : Prop where
  output : result.output=f.output
  finalState : result.finalState=f.finalState
  finalFlag : result.finalState.isSome=f.flag
  saved : result.saved=[f.q,f.k,f.v,f.history]
  launches : result.launches=forwardLaunches c
  launchDevice : result.launchDevice=f.q.device
  restoredDevice : result.restoredDevice=priorDevice
  admitted : HostABI.Accepted c
  values : RealForwardValues c (inputsOfHeap (forwardRealHeap c f before)) scale result
  stores : ∀ a∈Separate.forwardFootprint c f.flag,
    result.memory (cellAddress c f.tensor a)=Source.afterO realOps c scale tree f.flag (forwardRealHeap c f before) a
  frame : ∀ point,(∀ a∈Separate.forwardFootprint c f.flag,point≠cellAddress c f.tensor a) → result.memory point=before point
  caller : ∀ t∈callers,∀ point,t.storageBase≤point → point<t.storageBase+t.storageBytes → result.memory point=before point

structure BackwardRealPost (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (scale : ℝ) (tree : SumTree) (priorDevice : Nat) (before later : RealMemory)
    (callers : List Tensor) (result : RealBackwardReturn) : Prop where
  dq : result.dq=b.dq
  dk : result.dk=b.dk
  dv : result.dv=b.dv
  seedGradient : result.seedGradient=none
  flagGradient : result.flagGradient=none
  launches : result.launches=backwardLaunches c
  launchDevice : result.launchDevice=f.q.device
  restoredDevice : result.restoredDevice=priorDevice
  values : RealBackwardValues c (Separate.joined (forwardRealHeap c f before) (backwardRealHeap c f b later)) scale result
  vjp : RealVJP c (Separate.joined (forwardRealHeap c f before) (backwardRealHeap c f b later)) scale result
  stores : ∀ a∈Separate.backwardFootprint c,
    result.memory (cellAddress c b.tensor a)=Separate.backward realOps c scale tree (backwardRealHeap c f b later) a
  frame : ∀ point,(∀ a∈Separate.backwardFootprint c,point≠cellAddress c b.tensor a) →
    (b.incomingPresent=Bool.false → ¬(b.incoming.dataPointer≤point ∧ point<b.incoming.dataPointer+4*b.incoming.numel)) →
    result.memory point=later point
  caller : ∀ t∈callers,∀ point,t.storageBase≤point → point<t.storageBase+t.storageBytes → result.memory point=later point
  saved : ∀ t∈[f.q,f.k,f.v,f.history,f.output]++f.finalState.toList,∀ point,
    t.storageBase≤point → point<t.storageBase+t.storageBytes → result.memory point=later point
  incoming : b.incomingPresent=Bool.true → ∀ point,b.incoming.storageBase≤point →
    point<b.incoming.storageBase+b.incoming.storageBytes → result.memory point=later point
  terminal : ∀ t,b.terminal=some t → ∀ point,t.storageBase≤point →
    point<t.storageBase+t.storageBytes → result.memory point=later point

theorem forward_real (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (scale : ℝ) (tree : SumTree)
    (priorDevice : Nat) (before : RealMemory) (caller : List Tensor)
    (checks : forwardChecks (checkedShape g hs) f=Bool.true)
    (allocation : ForwardAllocation (checkedShape g hs) f caller) :
    ∃ result,forwardRealCall (checkedShape g hs) f scale tree priorDevice before=some result ∧
      ForwardRealPost (checkedShape g hs) f scale tree priorDevice before caller result := by
  have admit:=forward_store_admission (checkedShape g hs) f checks
  have hp : planChecks (checkedShape g hs)=Bool.true := by
    simp only [forwardChecks,Bool.and_eq_true] at checks
    exact checks.1.1
  refine ⟨realForwardReturn _ f scale tree priorDevice before,?_,rfl,rfl,f.finalFlag,rfl,rfl,rfl,rfl,
    plan_admitted g hs hp,forward_real_values _ f scale tree priorDevice before checks allocation.writes,?_,?_,?_⟩
  · simp [forwardRealCall,checks]
  · intro a ha
    exact finish_real_read _ f.tensor _ admit allocation.writes _ before a ha
  · intro point outside
    exact finish_real_frame _ _ _ _ _ point outside
  · intro t ht point hlo hhi
    exact finish_real_preserves _ f.tensor _ admit _ before t
      (fun a ha=>allocation.caller a ha t ht) point hlo hhi

theorem backward_real (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (b : BackwardObjects (checkedShape g hs) f)
    (scale : ℝ) (tree : SumTree) (complete : tree.Complete (checkedShape g hs).keyBlocks)
    (priorDevice : Nat) (before later : RealMemory) (fc bc : List Tensor)
    (fchecks : forwardChecks (checkedShape g hs) f=Bool.true)
    (bchecks : backwardChecks (checkedShape g hs) f b=Bool.true)
    (falloc : ForwardAllocation (checkedShape g hs) f fc) (balloc : BackwardAllocation (checkedShape g hs) f b bc)
    (saved : SavedReal (checkedShape g hs) f (forwardRealMemory (checkedShape g hs) f scale tree before) later) :
    ∃ result,backwardRealCall (checkedShape g hs) f b scale tree priorDevice later=some result ∧
      BackwardRealPost (checkedShape g hs) f b scale tree priorDevice before later bc result := by
  have admit:=backward_store_admission (checkedShape g hs) f b bchecks
  have sourceSaved:=saved_real_from_forward (checkedShape g hs) f b scale tree before later fc bc fchecks bchecks falloc balloc saved
  have values:=backward_real_values _ f b scale tree complete priorDevice before later bchecks balloc.writes sourceSaved
  refine ⟨realBackwardReturn _ f b scale tree priorDevice later,?_,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,values,
    real_vjp values,?_,?_,?_,?_,?_,?_⟩
  · simp [backwardRealCall,bchecks]
  · intro a ha
    exact finish_real_read _ b.tensor _ admit balloc.writes _ _ a ha
  · intro point outside zeroOutside
    change finishReal _ _ _ _ _ point=later point
    rw [finish_real_frame _ _ _ _ _ point outside]
    cases hi:b.incomingPresent
    · simp only [prepareReal,hi,↓reduceIte]
      exact if_neg (zeroOutside hi)
    · simp only [prepareReal,hi,↓reduceIte]
  · intro t ht point hlo hhi
    exact backward_real_preserves _ f b scale tree later bchecks t
      (fun a ha=>balloc.caller a ha t ht) (fun hn=>balloc.zeroCaller hn t ht) point hlo hhi
  · intro t ht point hlo hhi
    exact backward_real_preserves _ f b scale tree later bchecks t
      (fun a ha=>balloc.saved a ha t ht) (fun hn=>balloc.zeroSaved hn t ht) point hlo hhi
  · intro hi point hlo hhi
    exact backward_real_preserves _ f b scale tree later bchecks b.incoming (balloc.incoming hi)
      (fun hn=>False.elim (Bool.noConfusion (hi.symm.trans hn))) point hlo hhi
  · intro t ht point hlo hhi
    exact backward_real_preserves _ f b scale tree later bchecks t (balloc.terminal t ht)
      (fun hn=>balloc.zeroTerminal hn t ht) point hlo hhi

/-- Canonical unrestricted-Real task endpoint. Only Q/K/V/history are preserved
between calls; output/final-state/seed may be changed before backward. All
returned values, detached VJP, source stores, and frames are proved per call. -/
theorem whole_real (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (scale : ℝ) (tree : SumTree)
    (complete : tree.Complete (checkedShape g hs).keyBlocks)
    (priorDevice : Nat) (before : RealMemory) (caller : List Tensor)
    (checks : forwardChecks (checkedShape g hs) f=Bool.true)
    (allocation : ForwardAllocation (checkedShape g hs) f caller) :
    ∃ result,forwardRealCall (checkedShape g hs) f scale tree priorDevice before=some result ∧
      ForwardRealPost (checkedShape g hs) f scale tree priorDevice before caller result ∧
      ∀ (b : BackwardObjects (checkedShape g hs) f) (later : RealMemory) (backwardDevice : Nat) (bc : List Tensor),
        backwardChecks (checkedShape g hs) f b=Bool.true →
        BackwardAllocation (checkedShape g hs) f b bc →
        SavedReal (checkedShape g hs) f result.memory later →
        ∃ gradients,backwardRealCall (checkedShape g hs) f b scale tree backwardDevice later=some gradients ∧
          BackwardRealPost (checkedShape g hs) f b scale tree backwardDevice before later bc gradients := by
  obtain ⟨result,hcall,hpost⟩:=forward_real g hs f scale tree priorDevice before caller checks allocation
  refine ⟨result,hcall,hpost,?_⟩
  intro b later backwardDevice bc bchecks balloc saved
  have hr : result=realForwardReturn _ f scale tree priorDevice before := by
    apply Option.some.inj
    simpa only [forwardRealCall,checks,↓reduceIte] using hcall.symm
  rw [hr] at saved
  exact backward_real g hs f b scale tree complete backwardDevice before later caller bc
    checks bchecks allocation balloc saved

/-- Real erasure of the wrapper's K ** -0.5 default host scale. The more general
whole_real theorem permits every scale and therefore includes this default. -/
def defaultRealScale (c : Shape) : ℝ := (Real.sqrt (c.K : ℝ))⁻¹

#axiomsClean forward_real
#axiomsClean backward_real
#axiomsClean whole_real
#specNonCircular ForwardRealPost avoiding [forwardRealCall,forward_real,whole_real]
#specNonCircular BackwardRealPost avoiding [backwardRealCall,backward_real,whole_real]
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
