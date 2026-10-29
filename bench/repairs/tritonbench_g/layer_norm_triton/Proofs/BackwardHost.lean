import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Host

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical

/-- Additional ordinary Torch calls used by this source. Zero initialization
is a library operation, with a zero-cell contract only on its own allocation. -/
structure APIExtra (api : Memory.API UInt32 Store UInt8) (neg : UInt32 → UInt32)
    (inputs : List Storage.Tensor) (Metadata : Type) where
  zerosLike : Dense → Effect api neg inputs Metadata Dense
  zeroResult : ∀ t s u,((zerosLike t).call.run s).2=.ok u →
    AllocationResult (zerosLike t) s u ∧ u.tensor.view.shape=t.tensor.view.shape ∧ u.owner=t.owner
  zeroNative : ∀ t s u,((zerosLike t).call.run s).2=.ok u → ∀ i : Fin u.count,
    Storage.logical api neg ((zerosLike t).call.run s).1.memory u.tensor i.val=0
  zeroReal : ∀ t s u,((zerosLike t).call.run s).2=.ok u → ∀ m i,i<u.count →
    Storage.realLogical (((zerosLike t).transition s).shadow m) u.tensor i=0
  empty : Nat → List Nat → Effect api neg inputs Metadata Dense
  emptyResult : ∀ owner shape s u,((empty owner shape).call.run s).2=.ok u →
    AllocationResult (empty owner shape) s u ∧ u.tensor.view.shape=shape ∧ u.owner=owner
  zeroOwner : ∀ t s,((zerosLike t).call.run s).1.owner=s.owner
  emptyOwner : ∀ owner shape s,((empty owner shape).call.run s).1.owner=s.owner

structure Saved (c : SavedContext) (b : Dense) : Prop where
  bias : c.bias=some b
  xCount : c.x.count=c.rows*c.columns
  wCount : c.w.count=c.columns
  bCount : b.count=c.columns
  rCount : c.r.count=c.rows
  muCount : c.mu.count=c.rows
  xShape : c.x.tensor.view.shape=[c.rows,c.columns]
  wOwner : c.w.owner=c.x.owner
  bOwner : b.owner=c.x.owner
  inputShape : c.inputShape.prod=c.x.count
  weightShape : c.weightShape.prod=c.w.count
  biasShape : ∃ shape,c.biasShape=some shape ∧ shape.prod=b.count
  biasDevice : ∃ owner,c.biasDevice=some owner
  settings : Compiler.settings c.columns=.ok (c.block,Geometry.warps c.block)

structure Base where
  (dy dx dw db : Dense)

def outputs (p : Base) : Fin 3 → Dense
  | ⟨0,_⟩ => p.dx | ⟨1,_⟩ => p.dw | ⟨2,_⟩ => p.db

structure BaseLayout (p : Base) (c : SavedContext) (inputs : List Storage.Tensor) : Prop where
  dxCount : p.dx.count=c.rows*c.columns
  dwCount : p.dw.count=c.columns
  dbCount : p.db.count=c.columns
  dyCount : p.dy.count=c.rows*c.columns
  dyOwner : p.dy.owner=c.x.owner
  owners : ∀ k,(outputs p k).owner=c.x.owner
  separated : ∀ k j,k≠j → Storage.Disjoint (outputs p k).tensor.backing (outputs p j).tensor.backing
  callerFresh : ∀ k old,old∈inputs → Storage.Disjoint (outputs p k).tensor.backing old.backing
  freshDY : ∀ k,Storage.Disjoint (outputs p k).tensor.backing p.dy.tensor.backing

variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor} {Metadata : Type}

def prepareBase (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (c : SavedContext) (b : Dense) (dy : Storage.Tensor) : Action (world (Metadata:=Metadata) api inputs) Base :=
  (materialize lib c.x.tensor.view.device dy [c.rows,c.columns]).bind fun yd =>
  (call (.allocate .x) (lib.emptyLike c.x).call).bind fun dx =>
  (call (.allocate .weight) (extra.zerosLike c.w).call).bind fun dw =>
  (call (.allocate .bias) (extra.zerosLike b).call).bind fun db =>
  Action.pure ⟨yd,dx,dw,db⟩

structure BaseRoute (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (c : SavedContext) (b : Dense) (dy : Storage.Tensor) (initial : State Store Metadata) where
  dyr : MaterializeRoute lib c.x.tensor.view.device dy [c.rows,c.columns] initial
  (dx dw db : Dense)
  dxOK : ((lib.emptyLike c.x).call.run dyr.final).2=.ok dx
  dwOK : ((extra.zerosLike c.w).call.run ((lib.emptyLike c.x).call.run dyr.final).1).2=.ok dw
  dbOK : ((extra.zerosLike b).call.run
    ((extra.zerosLike c.w).call.run ((lib.emptyLike c.x).call.run dyr.final).1).1).2=.ok db

variable {lib : APIHost api neg inputs Metadata} {extra : APIExtra api neg inputs Metadata}
  {c : SavedContext} {b : Dense} {dy : Storage.Tensor} {initial : State Store Metadata}

def BaseRoute.sX (r : BaseRoute lib extra c b dy initial) := ((lib.emptyLike c.x).call.run r.dyr.final).1
def BaseRoute.sW (r : BaseRoute lib extra c b dy initial) := ((extra.zerosLike c.w).call.run r.sX).1
def BaseRoute.final (r : BaseRoute lib extra c b dy initial) := ((extra.zerosLike b).call.run r.sW).1
def BaseRoute.data (r : BaseRoute lib extra c b dy initial) : Base := ⟨r.dyr.output,r.dx,r.dw,r.db⟩
def BaseRoute.tail (r : BaseRoute lib extra c b dy initial) : Transition api neg inputs r.dyr.final r.final :=
  ((lib.emptyLike c.x).transition r.dyr.final).then
    (((extra.zerosLike c.w).transition r.sX).then ((extra.zerosLike b).transition r.sW))
def BaseRoute.transition (r : BaseRoute lib extra c b dy initial) : Transition api neg inputs initial r.final :=
  r.dyr.transition.then r.tail

theorem base_route (out : Base) (ok : ((prepareBase lib extra c b dy).run initial).result=.ok out) :
    ∃ r : BaseRoute lib extra c b dy initial,r.data=out ∧
      ((prepareBase lib extra c b dy).run initial).state=r.final := by
  obtain ⟨yd,hy,hx,hsy,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨yr,hey,hstate⟩ := materialize_route yd hy
  subst yd
  rw [hstate] at hx hsy
  obtain ⟨dx,hdx,hw,hsx,_⟩ := bind_success _ _ _ _ hx
  obtain ⟨dw,hdw,hb,hsw,_⟩ := bind_success _ _ _ _ hw
  obtain ⟨db,hdb,hret,hsb,_⟩ := bind_success _ _ _ _ hb
  let r : BaseRoute lib extra c b dy initial :=
    ⟨yr,dx,dw,db,(call_ok _ _ _ _).mp hdx,(call_ok _ _ _ _).mp hdw,(call_ok _ _ _ _).mp hdb⟩
  have hout : out=r.data := by simpa [Action.pure,r,BaseRoute.data] using hret.symm
  subst out
  exact ⟨r,rfl,hsy.trans (hsx.trans (hsw.trans hsb))⟩

theorem base_layout (r : BaseRoute lib extra c b dy initial) (saved : Saved c b) : BaseLayout r.data c inputs := by
  have hy := materialize_metadata r.dyr
  have hx := lib.emptyLikeResult c.x r.dyr.final r.dx r.dxOK
  have hw := extra.zeroResult c.w r.sX r.dw r.dwOK
  have hb := extra.zeroResult b r.sW r.db r.dbOK
  have xp := hx.1.present
  have wp := hw.1.present
  have xw := hw.1.fresh _ xp
  have xb := hb.1.fresh _ ((extra.zerosLike c.w).transition r.sX |>.live _ xp)
  have wb := hb.1.fresh _ wp
  refine ⟨?_,?_,?_,?_,hy.2.2.1.trans c.x.tensor.owner,?_,?_,?_,?_⟩
  · change r.dx.count=_
    rw [r.dx.countEq,hx.2.1,←c.x.countEq,saved.xCount]
  · change r.dw.count=_
    rw [r.dw.countEq,hw.2.1,←c.w.countEq,saved.wCount]
  · change r.db.count=_
    rw [r.db.countEq,hb.2.1,←b.countEq,saved.bCount]
  · change r.dyr.output.count=_
    rw [r.dyr.output.countEq,hy.1]
    simp
  · intro k
    fin_cases k
    · exact hx.2.2
    · exact hw.2.2.trans saved.wOwner
    · exact hb.2.2.trans saved.bOwner
  · intro k j h
    fin_cases k <;> fin_cases j
    · contradiction
    · exact disjoint_symm _ _ xw
    · exact disjoint_symm _ _ xb
    · exact xw
    · contradiction
    · exact disjoint_symm _ _ wb
    · exact xb
    · exact wb
    · contradiction
  · intro k old ho
    fin_cases k
    · exact hx.1.callerFresh old ho
    · exact hw.1.callerFresh old ho
    · exact hb.1.callerFresh old ho
  · intro k
    fin_cases k
    · exact hx.1.fresh _ hy.2.2.2
    · exact hw.1.fresh _ ((lib.emptyLike c.x).transition r.dyr.final |>.live _ hy.2.2.2)
    · exact hb.1.fresh _ ((extra.zerosLike c.w).transition r.sX |>.live _
        ((lib.emptyLike c.x).transition r.dyr.final |>.live _ hy.2.2.2))

theorem base_present (r : BaseRoute lib extra c b dy initial) :
    r.final.live r.data.dy.tensor ∧ ∀ k,r.final.live (outputs r.data k).tensor := by
  refine ⟨r.tail.live _ (materialize_metadata r.dyr).2.2.2,?_⟩
  intro k
  fin_cases k
  · exact (extra.zerosLike b).transition r.sW |>.live _
      ((extra.zerosLike c.w).transition r.sX |>.live _
        (lib.emptyLikeResult c.x r.dyr.final r.dx r.dxOK).1.present)
  · exact (extra.zerosLike b).transition r.sW |>.live _ (extra.zeroResult c.w r.sX r.dw r.dwOK).1.present
  · exact (extra.zeroResult b r.sW r.db r.dbOK).1.present

theorem base_zero_real (r : BaseRoute lib extra c b dy initial) (m : Storage.World (Nat → ℝ)) :
    (∀ i,i<r.dw.count → Storage.realLogical (r.transition.shadow m) r.dw.tensor i=0) ∧
    (∀ i,i<r.db.count → Storage.realLogical (r.transition.shadow m) r.db.tensor i=0) := by
  constructor
  · intro i hi
    have h := (extra.zerosLike b).transition r.sW |>.realValues
      (((extra.zerosLike c.w).transition r.sX).shadow (((lib.emptyLike c.x).transition r.dyr.final).shadow (r.dyr.transition.shadow m)))
      r.dw.tensor (extra.zeroResult c.w r.sX r.dw r.dwOK).1.present i (by simpa [r.dw.countEq] using hi)
    exact h.trans (extra.zeroReal c.w r.sX r.dw r.dwOK _ i hi)
  · exact extra.zeroReal b r.sW r.db r.dbOK _

#axiomsClean base_route
#axiomsClean base_layout
#axiomsClean base_zero_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
