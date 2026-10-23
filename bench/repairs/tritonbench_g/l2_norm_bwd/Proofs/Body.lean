import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Normalize

namespace VeriTile.Bench.Repairs.L2NormBwd.Host
open VeriTile.Triton FP.Structural Physical
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical
set_option maxHeartbeats 10000000
set_option linter.unusedSimpArgs false

private theorem byte_outside (a b : Allocation) (sep : a.separate b) (key : Key)
    (inside : b.contains key) (owner address : Nat) (observed : a.covers owner address) : ¬key.covers owner address := by
  rcases inside with ⟨ko,lo,hi⟩
  rcases observed with ⟨ao,alo,ahi⟩
  rintro ⟨bo,blo,bhi⟩
  rcases sep with sep | sep | sep
  · exact sep (ao.symm.trans (bo.trans ko))
  · omega
  · omega

def expected {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (call : Call) (initial : World M) (N B : Nat) (eps : α) (r c : Nat) : α :=
  red.apply N B eps (fun j => logical ops neg initial.memory call.x (r*N+j))
    (fun j => logical ops neg initial.memory call.dy (r*N+j)) c

structure BodySpec {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (call : Call) (initial : World M) (q : Request) (eps : α) (after : World M) : Prop where
  values : ∀ r c,r<q.x.rows → c<q.x.cols → ops.read after.memory (q.dx.key r c)=
    expected ops neg red call initial q.x.cols q.block eps r c
  caller : Frame ops call.caller initial after
  dxLive : after.live q.dx.view.storage

private theorem flat_bound (rows cols r c : Nat) (hr : r<rows) (hc : c<cols) : r*cols+c<rows*cols := by
  have bound := Nat.mul_le_mul_right cols (show r+1≤rows by omega)
  simp only [Nat.add_mul,Nat.one_mul] at bound
  omega

private theorem copy_fresh {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {v out : View} {before after : World M} (h : Copy ops neg v out before after)
    (a : Allocation) (live : before.live a) (separate : v.storage.separate a) : out.storage.separate a := by
  rcases h.storage with same | fresh
  · rwa [same]
  · exact fresh a live

/-- Actual resolve handles, their own metadata, setup, and all rows are used.
No successful kernel execution or whole-output mapping is a premise. -/
theorem body_spec {M α β : Type} (ops : Ops M α β) (neg : Negation α) (red : Reduction α)
    (rt : Runtime M) (lib : Library ops neg rt) (call : Call) (initial before : World M)
    (q : Request) (eps : α) (norm : Normalized ops neg call initial q before)
    (domain : InputDomain call initial) :
    Ensures (fun _ after => BodySpec ops neg red call initial q eps after)
      (fun _ after => Frame ops call.caller initial after) (body ops red rt q eps before) := by
  unfold body
  have f1 := lib.frame (.resolve q.x) before call.caller (fun a ha => norm.caller.live a (domain.live a ha))
  have fy1 := lib.frame (.resolve q.x) before [q.dy.view.storage] (by simpa using norm.dyLive)
  cases h1 : rt.perform (.resolve q.x) before with
  | error why failed =>
    simp only [h1,Result.state] at f1
    exact norm.caller.trans f1
  | ok x s1 =>
    simp only [h1,Result.state] at f1 fy1
    simp only [Result.bind]
    obtain ⟨xr,xc,xneg,cx,mx⟩ := lib.resolve _ _ _ _ h1
    have f2 := lib.frame (.resolve q.dy) s1 call.caller (fun a ha => f1.live a (norm.caller.live a (domain.live a ha)))
    have fx2 := lib.frame (.resolve q.dy) s1 [x.view.storage] (by simpa using cx.live)
    cases h2 : rt.perform (.resolve q.dy) s1 with
    | error why failed =>
      simp only [h2,Result.state] at f2
      exact (norm.caller.trans f1).trans f2
    | ok dy s2 =>
      simp only [h2,Result.state] at f2 fx2
      simp only [Result.bind]
      obtain ⟨yr,yc,yneg,cy,my⟩ := lib.resolve _ _ _ _ h2
      let p : Layout := ⟨x,dy,q.dx,q.block⟩
      have valid : Valid p := {
        positive := by dsimp [p];rw [xc];exact norm.positive
        block := by dsimp [p];rw [xc];exact norm.block
        dyCols := by dsimp [p];rw [yc,xc];exact norm.dyCols
        dyRows := by dsimp [p];rw [yr,xr];exact norm.dyRows
        dxCols := by dsimp [p];rw [xc];exact norm.dxCols
        dxRows := by dsimp [p];rw [xr];exact norm.dxRows
        xBacked := x.from_backed cx.backed
        dyBacked := dy.from_backed cy.backed
        dxBacked := q.dx.from_backed norm.dxBacked
        dxUnique := norm.dxUnique
        xFresh := copy_fresh cx q.dx.view.storage norm.dxLive norm.xFresh
        dyFresh := copy_fresh cy q.dx.view.storage (f1.live _ norm.dxLive) norm.dyFresh
        xResolved := xneg
        dyResolved := yneg
        dxResolved := norm.dxPositive }
      have f3 := lib.frame (.setup p s2.queue) s2 call.caller (fun a ha => f2.live a (f1.live a (norm.caller.live a (domain.live a ha))))
      cases h3 : rt.perform (.setup p s2.queue) s2 with
      | error why failed =>
        simp only [h3,Result.state] at f3
        exact ((norm.caller.trans f1).trans f2).trans f3
      | ok unit s3 =>
        cases unit
        have same := lib.setup _ _ _ _ h3
        subst s3
        simp only [Result.bind,Ensures]
        have schedule := lib.schedule p s2
        have bounded : ∀ r∈rt.schedule p s2,r<p.x.rows := by
          intro r hr;exact List.mem_range.mp (schedule.mem_iff.mp hr)
        have frame : Frame ops call.caller s2
            {s2 with memory:=Physical.run ops red p eps (rt.schedule p s2) s2.memory} := by
          refine ⟨fun _ h => h,?_,?_,rfl⟩
          · intro a ha key hk
            apply Physical.run_frame
            intro r hr c hc
            exact Allocation.spans_separate _ _ (norm.callerFresh a ha) _ _ hk
              (valid.dxBacked r c (by rw [valid.dxRows];exact bounded r hr) (by rw [valid.dxCols];exact hc))
          · intro a ha owner address hc
            apply Physical.run_bytes
            intro r hr c hi
            exact byte_outside a q.dx.view.storage (norm.callerFresh a ha) _
              (valid.dxBacked r c (by rw [valid.dxRows];exact bounded r hr) (by rw [valid.dxCols];exact hi)) owner address hc
        refine ⟨?_,((norm.caller.trans f1).trans f2).trans frame,f2.live _ (f1.live _ norm.dxLive)⟩
        intro r c hr hc
        have rp : r<p.x.rows := by dsimp [p];rwa [xr]
        have cp : c<p.x.cols := by dsimp [p];rwa [xc]
        rw [Physical.run_values ops red p valid eps (rt.schedule p s2)
          (schedule.nodup_iff.mpr List.nodup_range) bounded s2.memory r c
          (schedule.mem_iff.mpr (List.mem_range.mpr rp)) cp]
        dsimp [Physical.value,p,expected]
        rw [xc]
        apply red.congr q.x.cols q.block eps _ _ _ _ ?_ ?_ c hc
        · intro j hj
          have flat : r*q.x.cols+j<q.x.view.numel := by rw [Matrix.numel];exact flat_bound _ _ _ _ hr hj
          have flatX : r*q.x.cols+j<x.view.numel := by rw [cx.size];exact flat
          have active : r*q.x.cols+j<call.x.numel := by rw [←norm.xSize];exact flat
          have localKey : x.view.key (r*q.x.cols+j)=x.key r j := by rw [←xc];exact x.logical_key r j (by rwa [xc])
          have raw : inputs ops s2.memory x r j=logical ops neg s2.memory x.view (r*q.x.cols+j) := by
            simp [inputs,VeriTile.Bench.Repairs.SwigluTriton.Memory.logical,Negation.encode,xneg,localKey]
          rw [raw,fx2.logical neg x.view (by simp) cx.backed _ flatX,cx.values _ flat,norm.xValues _ active]
        · intro j hj
          have flat : r*q.x.cols+j<q.dy.view.numel := by
            rw [Matrix.numel,norm.dyCols]
            exact flat_bound _ _ _ _ (lt_of_lt_of_le hr norm.dyRows) hj
          have flatY : r*q.x.cols+j<dy.view.numel := by rw [cy.size];exact flat
          have active : r*q.x.cols+j<call.dy.numel := by rw [←norm.dySize];exact flat
          have localKey : dy.view.key (r*q.x.cols+j)=dy.key r j := by
            rw [←norm.dyCols,←yc];exact dy.logical_key r j (by rwa [yc,norm.dyCols])
          have raw : inputs ops s2.memory dy r j=logical ops neg s2.memory dy.view (r*q.x.cols+j) := by
            simp [inputs,VeriTile.Bench.Repairs.SwigluTriton.Memory.logical,Negation.encode,yneg,localKey]
          rw [raw,cy.values _ flat,fy1.logical neg q.dy.view (by simp) norm.dyBacked _ flat,norm.dyValues _ active]

#axiomsClean body_spec
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormBwd.Host
