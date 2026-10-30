import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Runtime

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 10000000

private theorem separate_symm (a b : Allocation) (h : a.separate b) : b.separate a := by
  rcases h with h | h | h
  · exact Or.inl (Ne.symm h)
  · exact Or.inr (Or.inr h)
  · exact Or.inr (Or.inl h)

theorem Copy.refl {M α β : Type} (ops : Ops M α β) (neg : Negation α) (v : View) (s : World M)
    (backed : v.backed) (live : s.live v.storage) : Copy ops neg v v s s :=
  ⟨rfl,rfl,backed,live,fun _ _ => rfl,Or.inl rfl⟩

theorem prepare_frame {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M)
    (lib : Library ops neg rt) (dy : Matrix) (s : World M) (kept : List Allocation)
    (live : ∀ old∈kept,s.live old) : Frame ops kept s (prepareDY rt dy s).state := by
  exact Frame.refl ops kept s

theorem prepare_facts {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M)
    (lib : Library ops neg rt) (dy : Matrix) (before after : World M) (out : Matrix)
    (success : prepareDY rt dy before=.ok out after)
    (backed : dy.view.backed) (live : before.live dy.view.storage) (metadata : Integer.Metadata dy) :
    out.rows=dy.rows ∧ out.cols=dy.cols ∧ Copy ops neg dy.view out.view before after ∧ Integer.Metadata out := by
  obtain ⟨rfl,rfl⟩ := Result.ok.inj success
  exact ⟨rfl,rfl,Copy.refl ops neg _ _ backed live,metadata⟩

structure Normalized {M α β : Type} (ops : Ops M α β) (neg : Negation α)
    (call : Call) (before : World M) (q : Request) (after : World M) : Prop where
  shape : q.originalShape=call.x.shape
  cols : q.x.cols=call.cols
  positive : 0<q.x.cols
  block : q.x.cols≤q.block
  blockBound : q.block≤16384
  blockActual : q.block=blockSize q.x.cols
  dyCols : q.dy.cols=q.x.cols
  dyRows : q.x.rows≤q.dy.rows
  dxRows : q.dx.rows=q.x.rows
  dxCols : q.dx.cols=q.x.cols
  xSize : q.x.view.numel=call.x.numel
  dySize : q.dy.view.numel=call.dy.numel
  xBacked : q.x.view.backed
  dyBacked : q.dy.view.backed
  dxBacked : q.dx.view.backed
  dxUnique : q.dx.injective
  dxPositive : q.dx.view.negative=Bool.false
  xLive : after.live q.x.view.storage
  dyLive : after.live q.dy.view.storage
  dxLive : after.live q.dx.view.storage
  xFresh : q.x.view.storage.separate q.dx.view.storage
  dyFresh : q.dy.view.storage.separate q.dx.view.storage
  callerFresh : ∀ old∈call.caller,old.separate q.dx.view.storage
  xValues : ∀ i,i<call.x.numel → logical ops neg after.memory q.x.view i=logical ops neg before.memory call.x i
  dyValues : ∀ i,i<call.dy.numel → logical ops neg after.memory q.dy.view i=logical ops neg before.memory call.dy i
  caller : Frame ops call.caller before after
  sameDevice : q.x.view.storage.owner=q.dy.view.storage.owner
  xMetadata : Integer.Metadata q.x
  dyMetadata : Integer.Metadata q.dy
  dxMetadata : Integer.Metadata q.dx
  dxDevice : q.dx.view.storage.owner=q.x.view.storage.owner

/-- This theorem inspects the actual sequence, including its failing state.
The result's dx is exactly the descriptor returned by this call's emptyLike. -/
theorem normalize_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α) (rt : Runtime M)
    (lib : Library ops neg rt) (call : Call) (s : World M) (domain : InputDomain call s) :
    Ensures (fun q after => Normalized ops neg call s q after)
      (fun _ after => Frame ops call.caller s after) (normalize rt call s) := by
  unfold normalize
  split
  · exact Frame.refl ops call.caller s
  ·
    have f1 := lib.frame (.flatten call.x call.cols) s call.caller domain.live
    cases h1 : rt.perform (.flatten call.x call.cols) s with
    | error why failed => simpa [h1,Result.bind,Ensures,Result.state] using f1
    | ok x s1 =>
      simp only [h1,Result.state] at f1
      simp only [Result.bind]
      obtain ⟨pos,nx,cx,mx⟩ := lib.flatten _ _ _ _ _ h1
      have f2 := lib.frame (.flatten call.dy call.cols) s1 call.caller (fun a ha => f1.live a (domain.live a ha))
      have fx2 := lib.frame (.flatten call.dy call.cols) s1 [x.view.storage] (by simpa using cx.live)
      cases h2 : rt.perform (.flatten call.dy call.cols) s1 with
      | error why failed =>
        simp only [h2,Result.state] at f2
        exact f1.trans f2
      | ok dy0 s2 =>
        simp only [h2,Result.state] at f2 fx2
        simp only [Result.bind]
        obtain ⟨_,ny,cy,my⟩ := lib.flatten _ _ _ _ _ h2
        have f3 := prepare_frame ops neg rt lib dy0 s2 call.caller (fun a ha => f2.live a (f1.live a (domain.live a ha)))
        have fx3 := prepare_frame ops neg rt lib dy0 s2 [x.view.storage] (by simpa using fx2.live _ cx.live)
        cases h3 : prepareDY rt dy0 s2 with
        | error why failed =>
          simp only [h3,Result.state] at f3
          exact (f1.trans f2).trans f3
        | ok dy s3 =>
          simp only [h3,Result.state] at f3 fx3
          simp only [Result.bind]
          obtain ⟨yr,yc,cd,md⟩ := prepare_facts ops neg rt lib dy0 s2 s3 dy h3 cy.backed cy.live my
          have f4 := lib.frame (.emptyLike x) s3 call.caller (fun a ha => f3.live a (f2.live a (f1.live a (domain.live a ha))))
          have fx4 := lib.frame (.emptyLike x) s3 [x.view.storage] (by simpa using fx3.live _ (fx2.live _ cx.live))
          have fy4 := lib.frame (.emptyLike x) s3 [dy.view.storage] (by simpa using cd.live)
          cases h4 : rt.perform (.emptyLike x) s3 with
          | error why failed =>
            simp only [h4,Result.state] at f4
            exact ((f1.trans f2).trans f3).trans f4
          | ok dx s4 =>
            simp only [h4,Result.state] at f4 fx4 fy4
            simp only [Result.bind]
            have frame := ((f1.trans f2).trans f3).trans f4
            obtain ⟨dr,dc,dneg,db,du,dl,df,dm,dxDevice⟩ := lib.emptyLike _ _ _ _ h4
            split
            · exact frame
            · rename_i width
              split
              · exact frame
              · rename_i enough
                split
                · exact frame
                · rename_i device
                  have xy : dy.cols=x.cols := yc.trans (ny.trans nx.symm)
                  have enoughRows : x.rows≤dy.rows := by
                    simp only [Matrix.numel,xy] at enough
                    exact Nat.le_of_mul_le_mul_right (Nat.le_of_not_gt enough) (by omega)
                  refine ⟨rfl,nx,(by change 0<x.cols;omega),(by change x.cols≤blockSize x.cols;omega),Nat.min_le_left _ _,rfl,xy,enoughRows,dr,dc,
                    cx.size,cd.size.trans cy.size,cx.backed,cd.backed,db,du,dneg,
                    fx4.live _ (fx3.live _ (fx2.live _ cx.live)),fy4.live _ cd.live,dl,
                    separate_symm _ _ (df _ (fx3.live _ (fx2.live _ cx.live))),
                    separate_symm _ _ (df _ cd.live),?_,?_,?_,frame,by simpa using device,mx,md,dm,dxDevice⟩
                  · intro old ho
                    exact separate_symm _ _ (df old (f3.live old (f2.live old (f1.live old (domain.live old ho)))))
                  · intro i hi
                    rw [(fx2.trans fx3 |>.trans fx4).logical neg x.view (by simp) cx.backed i (by rw [cx.size];exact hi),cx.values i hi]
                  · intro i hi
                    rw [fy4.logical neg dy.view (by simp) cd.backed i (by rw [cd.size,cy.size];exact hi),
                      cd.values i (by rw [cy.size];exact hi),cy.values i hi,
                      f1.logical neg call.dy (by simp [Call.caller]) domain.dyBacked i hi]

#axiomsClean normalize_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
