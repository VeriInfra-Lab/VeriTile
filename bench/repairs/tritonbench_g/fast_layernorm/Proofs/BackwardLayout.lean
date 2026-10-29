import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardPreparation
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Execution

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardLayout
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation
noncomputable section
open scoped Classical

def outputs (p : Data) : List Dense := [p.dx,p.dw]++p.db.toList

def affineCount (p : Data) : Nat := 1+(if p.db.isSome then 1 else 0)
def affineOutput (p : Data) (k : Fin (affineCount p)) : Dense :=
  if k.val=0 then p.dw else p.db.getD p.dw

structure Layout (c : SavedContext) (p : Data) (inputs : List Storage.Tensor) : Prop where
  saved : SavedLayout c
  dyCount : p.dy.count=c.rows*c.columns
  dxCount : p.dx.count=c.x.count
  dwCount : p.dw.count=c.w.count
  dbPresence : p.db.isSome=c.bias.isSome
  dbCount : ∀ b db,c.bias=some b → p.db=some db → db.count=b.count
  dyShape : p.dy.tensor.view.shape=[c.rows,c.columns]
  dxShape : p.dx.tensor.view.shape=[c.rows,c.columns]
  dyOwner : p.dy.owner=c.x.owner
  owners : ∀ out,out∈outputs p → out.owner=c.x.owner
  pairwise : (outputs p).Pairwise (fun a b => Storage.Disjoint a.tensor.backing b.tensor.backing)
  freshDy : ∀ out,out∈outputs p → Storage.Disjoint out.tensor.backing p.dy.tensor.backing
  callerFresh : ∀ out,out∈outputs p → ∀ old,old∈inputs → Storage.Disjoint out.tensor.backing old.backing

theorem affine_mem (p : Data) (k : Fin (affineCount p)) : affineOutput p k∈outputs p := by
  rcases p with ⟨dy,dx,dw,db⟩
  cases db <;> fin_cases k <;> simp [affineOutput,outputs]

theorem affine_separated (h : Layout c p inputs) (k j : Fin (affineCount p)) (hne : k≠j) :
    Storage.Disjoint (affineOutput p k).tensor.backing (affineOutput p j).tensor.backing := by
  rcases p with ⟨dy,dx,dw,db⟩
  cases db with
  | none => fin_cases k <;> fin_cases j; contradiction
  | some db =>
    have hp : Storage.Disjoint dw.tensor.backing db.tensor.backing := by
      have hp0 : ([dx,dw,db] : List Dense).Pairwise (fun a b => Storage.Disjoint a.tensor.backing b.tensor.backing) := h.pairwise
      have hp1 := (List.pairwise_cons.mp hp0).2
      exact (List.pairwise_cons.mp hp1).1 db (by simp)
    fin_cases k <;> fin_cases j
    · contradiction
    · exact hp
    · exact disjoint_symm _ _ hp
    · contradiction

def dxTrace (c : SavedContext) (ops : Ops α) (plan : Plan c.columns B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Layout c p inputs) (m : Storage.World Store) : Trace α 1 where
  owner := c.x.owner
  regions _ := backwardRegion ops plan api c.x c.w p.dy c.r c.mu p.dx m
  owned _ := layout.owners p.dx (by simp [outputs])
  separated k j h := False.elim (h (Subsingleton.elim _ _))

def affineTrace (c : SavedContext) (ops : Ops α) (plan : Plan c.columns B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Layout c p inputs) (m : Storage.World Store) : Trace α (affineCount p) where
  owner := c.x.owner
  regions k := ⟨affineOutput p k,fun i =>
    let value := affineReference ops plan c.rows (rows c.x c.columns m api)
      (rows p.dy c.columns m api) (Execution.saved api c.r c.mu m) i.val
    if k.val=0 then value.1 else value.2⟩
  owned k := layout.owners _ (affine_mem p k)
  separated := affine_separated layout

variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type} {lib : APIHost api neg inputs Metadata} {c : SavedContext} {dy : Storage.Tensor}
  {initial : State Store Metadata}

theorem optional_presence {b : Option Dense} (r : OptionalAllocation lib initial b) : r.output.isSome=b.isSome := by
  cases r <;> rfl

theorem optional_some {b : Option Dense} (r : OptionalAllocation lib initial b) (out : Dense)
    (h : r.output=some out) : ∃ original,b=some original ∧ ((lib.emptyLike original).call.run initial).2=.ok out := by
  cases r with
  | none => simp [OptionalAllocation.output] at h
  | @some original result ok =>
    have he : result=out := by simpa [OptionalAllocation.output] using h
    subst out
    exact ⟨original,rfl,ok⟩

theorem route_layout (r : Route lib c dy initial) (saved : SavedLayout c) : Layout c r.data inputs := by
  have hy := Preparation.materialize_metadata r.dyr
  have hx := lib.emptyLikeResult c.x r.dyr.final r.dx r.dxOK
  have hw := lib.emptyLikeResult c.w r.sX r.dw r.dwOK
  have dyPresent := hy.2.2.2
  have dxPresent := hx.1.present
  have dwPresent := hw.1.present
  have dxDy := hx.1.fresh _ dyPresent
  have dwDy := hw.1.fresh _ ((lib.emptyLike c.x).transition r.dyr.final |>.live _ dyPresent)
  have dwDx := hw.1.fresh _ dxPresent
  have xCount : r.dx.count=c.x.count := by rw [r.dx.countEq,hx.2.1,←c.x.countEq]
  have wCount : r.dw.count=c.w.count := by rw [r.dw.countEq,hw.2.1,←c.w.countEq]
  have dyCount : r.dyr.output.count=c.rows*c.columns := by rw [r.dyr.output.countEq,hy.1]; simp
  have dwOwner : r.dw.owner=c.x.owner := hw.2.2.trans (saved.owners c.w (by simp))
  have dyOwner : r.dyr.output.owner=c.x.owner := hy.2.2.1.trans c.x.tensor.owner
  refine ⟨saved,dyCount,xCount,wCount,optional_presence r.db,?_,hy.1,hx.2.1.trans saved.xShape,dyOwner,?_,?_,?_,?_⟩
  · intro b db hb hdb
    obtain ⟨original,horig,hok⟩ := optional_some r.db db hdb
    have he : original=b := Option.some.inj (horig.symm.trans hb)
    subst original
    rw [db.countEq,(lib.emptyLikeResult b r.sW db hok).2.1,←b.countEq]
  · intro out ho
    cases hbr : r.db.output with
    | none =>
      simp [outputs,Route.data,hbr] at ho
      rcases ho with rfl|rfl
      · exact hx.2.2
      · exact dwOwner
    | some db =>
      obtain ⟨b,hb,ok⟩ := optional_some r.db db hbr
      have hdb := lib.emptyLikeResult b r.sW db ok
      simp [outputs,Route.data,hbr] at ho
      rcases ho with rfl|rfl|rfl
      · exact hx.2.2
      · exact dwOwner
      · exact hdb.2.2.trans (saved.owners b (by simp [hb]))
  · cases hbr : r.db.output with
    | none => simpa [outputs,Route.data,hbr] using disjoint_symm _ _ dwDx
    | some db =>
      obtain ⟨b,hb,ok⟩ := optional_some r.db db hbr
      have hdb := lib.emptyLikeResult b r.sW db ok
      have dbDx := hdb.1.fresh _ ((lib.emptyLike c.w).transition r.sX |>.live _ dxPresent)
      have dbDw := hdb.1.fresh _ dwPresent
      simp [outputs,Route.data,hbr,List.pairwise_cons]
      exact ⟨⟨disjoint_symm _ _ dwDx,disjoint_symm _ _ dbDx⟩,disjoint_symm _ _ dbDw⟩
  · intro out ho
    cases hbr : r.db.output with
    | none =>
      simp [outputs,Route.data,hbr] at ho
      rcases ho with rfl|rfl
      · exact dxDy
      · exact dwDy
    | some db =>
      obtain ⟨b,hb,ok⟩ := optional_some r.db db hbr
      have hdb := lib.emptyLikeResult b r.sW db ok
      simp [outputs,Route.data,hbr] at ho
      rcases ho with rfl|rfl|rfl
      · exact dxDy
      · exact dwDy
      · exact hdb.1.fresh _ ((lib.emptyLike c.w).transition r.sX |>.live _
          ((lib.emptyLike c.x).transition r.dyr.final |>.live _ dyPresent))
  · intro out ho old hold
    cases hbr : r.db.output with
    | none =>
      simp [outputs,Route.data,hbr] at ho
      rcases ho with rfl|rfl
      · exact hx.1.callerFresh old hold
      · exact hw.1.callerFresh old hold
    | some db =>
      obtain ⟨b,hb,ok⟩ := optional_some r.db db hbr
      simp [outputs,Route.data,hbr] at ho
      rcases ho with rfl|rfl|rfl
      · exact hx.1.callerFresh old hold
      · exact hw.1.callerFresh old hold
      · exact (lib.emptyLikeResult b r.sW out ok).1.callerFresh old hold

#axiomsClean route_layout
#axiomsClean affine_separated
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardLayout
