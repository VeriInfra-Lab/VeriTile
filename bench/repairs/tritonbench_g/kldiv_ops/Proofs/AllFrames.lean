import bench.repairs.tritonbench_g.kldiv_ops.Proofs.HostBackward

namespace VeriTile.Bench.Repairs.KldivOps.Host
open Storage VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section
set_option maxHeartbeats 6000000

def resultState : Result A→State
  | .error _ s=>s
  | .ok _ s=>s

def AllPreserves (action : Action A) : Prop := ∀ s,Preserves s (resultState (action s))
def AllModified (dst : Tensor) (action : Action A) : Prop := ∀ s,Modified dst s (resultState (action s))

/-- Ordinary primitive frame contracts on success and failure. copy_ may
partially modify its logical destination on failure; no rollback is assumed.
No kernel result, indexing law, or successful execution is postulated here. -/
structure PrimitiveFrames {Word : Type} (api : Services Word) (compiler : Compiler) : Prop where
  conj : ∀ a,AllPreserves (api.resolveConj a)
  neg : ∀ a,AllPreserves (api.resolveNeg a)
  contiguous : ∀ a,AllPreserves (api.contiguous a)
  clone : ∀ a,AllPreserves (api.clone a)
  zeros : ∀ sh dev,AllPreserves (api.zeros sh dev)
  empty : ∀ a,AllPreserves (api.emptyContiguousLike a)
  copy : ∀ dst src,AllModified dst (api.copy dst src)
  sum : ∀ a dim0,AllPreserves (api.sum a dim0)
  divide : ∀ a n,AllPreserves (api.divNat a n)
  one : ∀ dev,AllPreserves (api.scalarOne dev)
  equal : ∀ a b,AllPreserves (api.equal a b)
  multiply : ∀ a b,AllPreserves (api.multiply a b)
  configure : ∀ rows cols,AllPreserves (api.configure rows cols)
  forwardSetup : ∀ rows cols p t o red,AllPreserves (compiler.forward rows cols p t o red)
  backwardSetup : ∀ rows cols t o,AllPreserves (compiler.backward rows cols t o)

theorem preserve_bind (first : Action A) (next : A→Action B)
    (h : AllPreserves first) (k : ∀ a,AllPreserves (next a)) : AllPreserves (bind first next) := by
  intro s
  cases call : first s with
  | error e t=>simpa [bind,call] using h s
  | ok a t=>
    have hs : Preserves s t := by simpa [call,resultState] using h s
    simpa [bind,call] using hs.trans (k a t)

theorem prepare_frames {Word : Type} {api : Services Word} {compiler : Compiler}
    (frames : PrimitiveFrames api compiler) (a : Tensor) : AllPreserves (prepare api a) := by
  exact preserve_bind _ _ (preserve_bind _ _ (frames.conj a) frames.neg) frames.contiguous

theorem snapshot_frames {Word : Type} {api : Services Word} {compiler : Compiler}
    (frames : PrimitiveFrames api compiler) (a : Tensor) : AllPreserves (snapshot api a) := by
  exact preserve_bind _ _ (preserve_bind _ _ (frames.conj a) frames.neg) frames.clone

theorem capacity_frames (a : Tensor) (count : Nat) (e : String) : AllPreserves (requireCapacity a count e) := by
  intro s
  unfold requireCapacity
  split <;> exact Preserves.refl s

theorem finish_forward_frames {Word : Type} {api : Services Word} {compiler : Compiler}
    (frames : PrimitiveFrames api compiler) (red : Reduction) (rows cols : Nat) (out : Tensor) :
    AllPreserves (finishForward api red rows cols out) := by
  cases red with
  | none=>exact fun s=>Preserves.refl s
  | sum=>exact frames.sum out Bool.true
  | mean=>exact preserve_bind _ _ (frames.sum out Bool.false) (fun a=>frames.divide a (rows*cols))
  | batchmean=>exact preserve_bind _ _ (frames.sum out Bool.false) (fun a=>frames.divide a rows)

theorem finish_backward_frames {Word : Type} {api : Services Word} {compiler : Compiler}
    (frames : PrimitiveFrames api compiler) (grad dst work : Tensor) : AllModified dst (backwardFinish api grad dst work) := by
  intro s
  cases hc : api.copy dst work s with
  | error e t=>simpa [backwardFinish,Bind.bind,bind,hc] using frames.copy dst work s
  | ok result t=>
    have copied : Modified dst s t := by simpa [hc,resultState] using frames.copy dst work s
    have rest : AllPreserves (do
        let one ← api.scalarOne grad.device
        let test ← api.equal grad one
        if test then pure dst else api.multiply dst grad) := by
      apply preserve_bind _ _ (frames.one grad.device)
      intro one
      apply preserve_bind _ _ (frames.equal grad one)
      intro test
      cases test with
      | «false»=>exact frames.multiply dst grad
      | «true»=>exact fun u=>Preserves.refl u
    simpa [backwardFinish,Bind.bind,bind,hc] using copied.trans ((rest t).modified dst)

/-- Any successful or failed full forward call preserves every entry byte.
The fresh output writes are derived from the full typed grid. -/
theorem forward_all_frames {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (lib : Library codec num api) (compiler : Compiler)
    (frames : PrimitiveFrames api compiler) (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word) :
    AllPreserves (forward api (sourceActions codec M compiler) prediction target flag name eps) := by
  intro entry
  cases hs : shape2 prediction entry with
  | error e s=>
    have eq : s=entry := by unfold shape2 at hs; split at hs <;> cases hs <;> rfl
    subst s
    simp [forward,Bind.bind,bind,hs,resultState,Preserves.refl]
  | ok dims s=>
    have eq := (shape2_success hs).2
    subst s
    cases hc : api.configure dims.1 dims.2 entry with
    | error e s=>simpa [forward,Bind.bind,bind,hs,hc] using frames.configure dims.1 dims.2 entry
    | ok unit s=>
      cases unit
      have pc : Preserves entry s := by simpa [hc,resultState] using frames.configure dims.1 dims.2 entry
      cases hr : lookup name s with
      | error e t=>
        have eq : t=s := by
          unfold lookup at hr
          split at hr <;> try cases hr
          split at hr <;> try cases hr
          split at hr <;> try cases hr
          split at hr <;> cases hr <;> rfl
        subst t
        simpa [forward,Bind.bind,bind,hs,hc,hr,resultState] using pc
      | ok red t=>
        have eq := lookup_state hr
        subst t
        cases ha : api.zeros (red.outputShape dims.1 dims.2) prediction.device s with
        | error e t=>
          have pa := frames.zeros (red.outputShape dims.1 dims.2) prediction.device s
          simpa [forward,Bind.bind,bind,hs,hc,hr,ha,resultState] using pc.trans pa
        | ok out allocated=>
          have alloc := (lib.zeros _ _ _ _ _ ha).1
          have pa := pc.trans alloc.preserved
          cases hy : prepare api prediction allocated with
          | error e t=>
            have py := prepare_frames frames prediction allocated
            simpa [forward,Bind.bind,bind,hs,hc,hr,ha,hy,resultState] using pa.trans py
          | ok py packed=>
            have pp : Preserves entry packed := pa.trans (by simpa [hy,resultState] using prepare_frames frames prediction allocated)
            cases ht : prepare api target packed with
            | error e t=>
              have pt := prepare_frames frames target packed
              simpa [forward,Bind.bind,bind,hs,hc,hr,ha,hy,ht,resultState] using pp.trans pt
            | ok pt ready=>
              have pr : Preserves entry ready := pp.trans (by simpa [ht,resultState] using prepare_frames frames target packed)
              cases hv : requireCapacity pt (dims.1*dims.2) "target has fewer elements than prediction" ready with
              | error e t=>
                have pv := capacity_frames pt (dims.1*dims.2) "target has fewer elements than prediction" ready
                simpa [forward,Bind.bind,bind,hs,hc,hr,ha,hy,ht,hv,resultState] using pr.trans pv
              | ok unit checked=>
                cases unit
                have eq := (capacity_success hv).2
                subst checked
                cases hl : launchForward codec M compiler dims.1 dims.2 py pt out flag red eps ready with
                | error e t=>
                  have failed := forward_failure_is_setup codec M compiler _ _ _ _ _ _ _ _ _ _ _ hl
                  have pf := frames.forwardSetup dims.1 dims.2 py pt out red ready
                  simpa [forward,Bind.bind,bind,hs,hc,hr,ha,hy,ht,hv,sourceActions,hl,resultState,failed] using pr.trans pf
                | ok unit launched=>
                  cases unit
                  obtain ⟨data⟩ := (by simpa [hl,Ensures] using forward_complete codec M compiler dims.1 dims.2 py pt out flag red eps ready :
                    Nonempty (ForwardData codec M compiler dims.1 dims.2 py pt out flag red eps ready launched))
                  have pl : Preserves entry launched := ⟨by simpa [data.live] using pr.live,
                    fun b hb=>(pr.frame b hb).trans (data.frame b (alloc.fresh b (pc.live b hb)))⟩
                  have pf := finish_forward_frames frames red dims.1 dims.2 out launched
                  change Preserves entry (resultState (forward api (sourceActions codec M compiler) prediction target flag name eps entry))
                  simpa [forward,Bind.bind,bind,hs,hc,hr,ha,hy,ht,hv,sourceActions,hl,finishForward] using pl.trans pf

/-- Full backward preserves every entry byte outside the supplied logical
write destination, on success and each failure. Gaps and aliased views are
included because Written is a physical byte predicate. -/
theorem backward_all_frames {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (lib : Library codec num api) (compiler : Compiler)
    (frames : PrimitiveFrames api compiler) (target grad dst : Tensor) (flag : Bool)
    (entry : State) (dlive : dst.backing∈entry.live) :
    Modified dst entry (resultState (backward api (sourceActions codec M compiler) target grad dst flag entry)) := by
  cases hs : shape2 target entry with
  | error e s=>
    have eq : s=entry := by unfold shape2 at hs; split at hs <;> cases hs <;> rfl
    subst s
    simpa [backward,Bind.bind,bind,hs,resultState] using (Preserves.refl entry).modified dst
  | ok dims s=>
    have eq := (shape2_success hs).2
    subst s
    cases hc : api.configure dims.1 dims.2 entry with
    | error e s=>simpa [backward,Bind.bind,bind,hs,hc] using (frames.configure dims.1 dims.2 entry).modified dst
    | ok unit s=>
      cases unit
      have pc : Preserves entry s := by simpa [hc,resultState] using frames.configure dims.1 dims.2 entry
      cases ht : prepare api target s with
      | error e t=>
        have pt := prepare_frames frames target s
        simpa [backward,Bind.bind,bind,hs,hc,ht,resultState] using (pc.trans pt).modified dst
      | ok pt packed=>
        have pp : Preserves entry packed := pc.trans (by simpa [ht,resultState] using prepare_frames frames target s)
        cases hg : snapshot api grad packed with
        | error e t=>
          have pg := snapshot_frames frames grad packed
          simpa [backward,Bind.bind,bind,hs,hc,ht,hg,resultState] using (pp.trans pg).modified dst
        | ok pg cloned=>
          have pcl : Preserves entry cloned := pp.trans (by simpa [hg,resultState] using snapshot_frames frames grad packed)
          cases ha : api.emptyContiguousLike dst cloned with
          | error e t=>
            have pa := frames.empty dst cloned
            simpa [backward,Bind.bind,bind,hs,hc,ht,hg,ha,resultState] using (pcl.trans pa).modified dst
          | ok work ready=>
            have alloc := lib.empty dst cloned work ready (pcl.live _ dlive) ha
            have pr := pcl.trans alloc.preserved
            cases hv : requireCapacity work (dims.1*dims.2) "new_grads has fewer elements than target" ready with
            | error e t=>
              have pv := capacity_frames work (dims.1*dims.2) "new_grads has fewer elements than target" ready
              simpa [backward,Bind.bind,bind,hs,hc,ht,hg,ha,hv,resultState] using (pr.trans pv).modified dst
            | ok unit checked=>
              cases unit
              have eq := (capacity_success hv).2
              subst checked
              cases hl : launchBackward codec M compiler dims.1 dims.2 pt work flag ready with
              | error e t=>
                have failed := backward_failure_is_setup codec M compiler _ _ _ _ _ _ _ _ hl
                have pf := frames.backwardSetup dims.1 dims.2 pt work ready
                simpa [backward,Bind.bind,bind,hs,hc,ht,hg,ha,hv,sourceActions,hl,resultState,failed] using (pr.trans pf).modified dst
              | ok unit launched=>
                cases unit
                obtain ⟨data⟩ := (by simpa [hl,Ensures] using backward_complete codec M compiler dims.1 dims.2 pt work flag ready :
                  Nonempty (BackwardData codec M compiler dims.1 dims.2 pt work flag ready launched))
                have pl : Preserves entry launched := ⟨by simpa [data.live] using pr.live,
                  fun b hb=>(pr.frame b hb).trans (data.frame b (alloc.fresh b (pcl.live b hb)))⟩
                have pf := finish_backward_frames frames pg dst work launched
                simpa [backward,Bind.bind,bind,hs,hc,ht,hg,ha,hv,sourceActions,hl] using (pl.modified dst).trans pf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Host
