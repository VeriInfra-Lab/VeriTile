import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardHost

namespace VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation ReceiptAction
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor} {Metadata : Type}

structure Scratch where
  (dw db : Dense)

def scratchShape (c : SavedContext) : List Nat := [min (Geometry.groups c.columns) c.rows,c.columns]

def prepareScratch (extra : APIExtra api neg inputs Metadata) (c : SavedContext) :
    Action (world (Metadata:=Metadata) api inputs) Scratch :=
  (call (.allocate .weight) (extra.empty c.x.owner (scratchShape c)).call).bind fun dw =>
  (call (.allocate .bias) (extra.empty c.x.owner (scratchShape c)).call).bind fun db =>
  Action.pure ⟨dw,db⟩

structure ScratchRoute (extra : APIExtra api neg inputs Metadata) (c : SavedContext) (initial : State Store Metadata) where
  (dw db : Dense)
  dwOK : ((extra.empty c.x.owner (scratchShape c)).call.run initial).2=.ok dw
  dbOK : ((extra.empty c.x.owner (scratchShape c)).call.run
    ((extra.empty c.x.owner (scratchShape c)).call.run initial).1).2=.ok db

variable {extra : APIExtra api neg inputs Metadata} {c : SavedContext} {initial : State Store Metadata}
def ScratchRoute.middle (r : ScratchRoute extra c initial) := ((extra.empty c.x.owner (scratchShape c)).call.run initial).1
def ScratchRoute.final (r : ScratchRoute extra c initial) := ((extra.empty c.x.owner (scratchShape c)).call.run r.middle).1
def ScratchRoute.data (r : ScratchRoute extra c initial) : Scratch := ⟨r.dw,r.db⟩
def ScratchRoute.transition (r : ScratchRoute extra c initial) : Transition api neg inputs initial r.final :=
  ((extra.empty c.x.owner (scratchShape c)).transition initial).then ((extra.empty c.x.owner (scratchShape c)).transition r.middle)

theorem scratch_route (out : Scratch) (ok : ((prepareScratch extra c).run initial).result=.ok out) :
    ∃ r : ScratchRoute extra c initial,r.data=out ∧ ((prepareScratch extra c).run initial).state=r.final := by
  obtain ⟨dw,hw,hb,hsw,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨db,hdb,hret,hsb,_⟩ := bind_success _ _ _ _ hb
  let r : ScratchRoute extra c initial := ⟨dw,db,(call_ok _ _ _ _).mp hw,(call_ok _ _ _ _).mp hdb⟩
  have hout : out=r.data := by simpa [Action.pure,r,ScratchRoute.data] using hret.symm
  subst out
  exact ⟨r,rfl,hsw.trans hsb⟩

def groupOutputs (base : Base) (scratch : Scratch) : Fin 3 → Dense
  | ⟨0,_⟩ => base.dx | ⟨1,_⟩ => scratch.dw | ⟨2,_⟩ => scratch.db

structure ScratchLayout (base : Base) (scratch : Scratch) (c : SavedContext) (inputs : List Storage.Tensor) : Prop where
  dwCount : scratch.dw.count=min (Geometry.groups c.columns) c.rows*c.columns
  dbCount : scratch.db.count=min (Geometry.groups c.columns) c.rows*c.columns
  owners : ∀ k,(groupOutputs base scratch k).owner=c.x.owner
  separated : ∀ k j,k≠j → Storage.Disjoint (groupOutputs base scratch k).tensor.backing (groupOutputs base scratch j).tensor.backing
  callerFresh : ∀ k old,old∈inputs → Storage.Disjoint (groupOutputs base scratch k).tensor.backing old.backing

theorem scratch_layout (base : Base) (layout : BaseLayout base c inputs)
    (present : initial.live base.dx.tensor) (r : ScratchRoute extra c initial) : ScratchLayout base r.data c inputs := by
  have hw := extra.emptyResult c.x.owner (scratchShape c) initial r.dw r.dwOK
  have hb := extra.emptyResult c.x.owner (scratchShape c) r.middle r.db r.dbOK
  have wx := hw.1.fresh _ present
  have bx := hb.1.fresh _ ((extra.empty c.x.owner (scratchShape c)).transition initial |>.live _ present)
  have bw := hb.1.fresh _ hw.1.present
  refine ⟨?_,?_,?_,?_,?_⟩
  · change r.dw.count=_
    simp [r.dw.countEq,hw.2.1,scratchShape]
  · change r.db.count=_
    simp [r.db.countEq,hb.2.1,scratchShape]
  · intro k
    fin_cases k
    · exact layout.owners 0
    · exact hw.2.2
    · exact hb.2.2
  · intro k j h
    fin_cases k <;> fin_cases j
    · contradiction
    · exact disjoint_symm _ _ wx
    · exact disjoint_symm _ _ bx
    · exact wx
    · contradiction
    · exact disjoint_symm _ _ bw
    · exact bx
    · exact bw
    · contradiction
  · intro k old ho
    fin_cases k
    · exact layout.callerFresh 0 old ho
    · exact hw.1.callerFresh old ho
    · exact hb.1.callerFresh old ho

def groupBuffers (base : Base) (scratch : Scratch) (layout : BaseLayout base c inputs)
    (sl : ScratchLayout base scratch c inputs) : Execution.BackwardBuffers c.rows c.columns (Geometry.groups c.columns) inputs :=
  ⟨⟨c.x.owner,groupOutputs base scratch,sl.owners,sl.separated,sl.callerFresh⟩,layout.dxCount,sl.dwCount,sl.dbCount⟩

def finalOutputs (base : Base) : Fin 2 → Dense | ⟨0,_⟩ => base.dw | ⟨1,_⟩ => base.db

def finalBuffers (base : Base) (layout : BaseLayout base c inputs) : Execution.FinalBuffers c.columns inputs :=
  ⟨⟨c.x.owner,finalOutputs base,
    by intro k; fin_cases k; exact layout.owners 1; exact layout.owners 2,
    by
      intro k j h
      fin_cases k <;> fin_cases j
      · contradiction
      · exact layout.separated 1 2 (by decide)
      · exact layout.separated 2 1 (by decide)
      · contradiction,
    by intro k old ho; fin_cases k; exact layout.callerFresh 1 old ho; exact layout.callerFresh 2 old ho⟩,
    layout.dwCount,layout.dbCount⟩

structure All where
  base : Base
  scratch : Option Scratch

structure Ready (data : All) (c : SavedContext) (inputs : List Storage.Tensor) : Prop where
  base : BaseLayout data.base c inputs
  scratch : ∀ s,data.scratch=some s → ScratchLayout data.base s c inputs

def prepareAll (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (c : SavedContext) (b : Dense) (dy : Storage.Tensor) : Action (world (Metadata:=Metadata) api inputs) All :=
  (prepareBase lib extra c b dy).bind fun base =>
  if c.rows≠0 ∧ c.columns≠0 then
    (prepareScratch extra c).bind fun scratch => Action.pure ⟨base,some scratch⟩
  else Action.pure ⟨base,none⟩

theorem prepareAll_ready (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (c : SavedContext) (b : Dense) (saved : Saved c b) (dy : Storage.Tensor)
    (initial : State Store Metadata) (out : All) (ok : ((prepareAll lib extra c b dy).run initial).result=.ok out) :
    Ready out c inputs := by
  obtain ⟨base,hb,rest,hs,_⟩ := bind_success _ _ _ _ ok
  obtain ⟨br,he,hstate⟩ := base_route base hb
  subst base
  rw [hstate] at rest
  split at rest
  · obtain ⟨s,hs,hret,_,_⟩ := bind_success _ _ _ _ rest
    obtain ⟨sr,he,_⟩ := scratch_route s hs
    subst s
    have hout : out=⟨br.data,some sr.data⟩ := by simpa [Action.pure] using hret.symm
    subst out
    refine ⟨base_layout br saved,?_⟩
    intro s he
    cases he
    exact scratch_layout br.data (base_layout br saved) ((base_present br).2 0) sr
  · have hout : out=⟨br.data,none⟩ := by simpa [Action.pure] using rest.symm
    subst out
    exact ⟨base_layout br saved,by intro s hs; cases hs⟩

def prepared (lib : APIHost api neg inputs Metadata) (extra : APIExtra api neg inputs Metadata)
    (c : SavedContext) (b : Dense) (saved : Saved c b) (dy : Storage.Tensor) :
    Action (world (Metadata:=Metadata) api inputs) {out : All // Ready out c inputs} :=
  certify (prepareAll lib extra c b dy) _ (prepareAll_ready lib extra c b saved dy)

#axiomsClean scratch_layout
#axiomsClean prepareAll_ready
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.BackwardHost
