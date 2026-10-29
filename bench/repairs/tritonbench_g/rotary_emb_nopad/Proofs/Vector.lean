import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Grid
import bench.repairs.tritonbench_g.rotary_emb.Proofs.Vector

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
open Checked
variable {α : Type}

def vectorWrites (c : Checked.Call) (indices : Indices) (p : Ops α) (ws : List Work) (m : Memory α) : List (VeriTile.Bench.Repairs.RotaryEmb.Stores.Write α) :=
  let output := fun (odd : Bool) (w : Work) =>
    ((.other c.request.device,destination c.request indices w odd),component (sourceValues p c.request w m) odd)
  ws.map (output false) ++ ws.map (output true)

/-- A source branch: all four loads and both output expressions precede the
even scatter; the odd scatter uses those same loaded values. -/
def vectorBranch (c : Checked.Call) (indices : Indices) (p : Ops α) (ws : List Work) (m : Memory α) : Memory α :=
  VeriTile.Bench.Repairs.RotaryEmb.Stores.commit (vectorWrites c indices p ws m) m

theorem vectorWrites_mem (c : Checked.Call) (indices : Indices) (p : Ops α) (ws : List Work) (m : Memory α)
    (x : VeriTile.Bench.Repairs.RotaryEmb.Stores.Write α) : x∈vectorWrites c indices p ws m ↔
      ∃ w∈ws,∃ odd,x=((.other c.request.device,destination c.request indices w odd),component (sourceValues p c.request w m) odd) := by
  constructor
  · intro h
    rcases List.mem_append.mp h with h|h
    · rcases List.mem_map.mp h with ⟨w,hw,rfl⟩
      exact ⟨w,hw,false,rfl⟩
    · rcases List.mem_map.mp h with ⟨w,hw,rfl⟩
      exact ⟨w,hw,true,rfl⟩
  · rintro ⟨w,hw,odd,rfl⟩
    change _∈_++_
    cases odd
    · exact List.mem_append_left _ (List.mem_map.mpr ⟨w,hw,rfl⟩)
    · exact List.mem_append_right _ (List.mem_map.mpr ⟨w,hw,rfl⟩)

theorem vector_coherent (c : Checked.Call) (indices : Indices) (accepted : Checked.check c indices=true) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.request.Active w) (m : Memory α) : VeriTile.Bench.Repairs.RotaryEmb.Stores.Coherent (vectorWrites c indices p ws m) := by
  intro a ha b hb equal
  rcases (vectorWrites_mem c indices p ws m a).mp ha with ⟨w,hw,ow,rfl⟩
  rcases (vectorWrites_mem c indices p ws m b).mp hb with ⟨v,hv,ov,rfl⟩
  have e := Prod.mk.inj equal
  obtain ⟨rfl,rfl⟩ := destination_injective c indices accepted w v (active w hw) (active v hv) ow ov e.2
  rfl

theorem vector_readback (c : Checked.Call) (indices : Indices) (accepted : Checked.check c indices=true) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.request.Active w) (m : Memory α) (w : Work) (hw : w∈ws) (odd : Bool) :
    vectorBranch c indices p ws m (.other c.request.device) (destination c.request indices w odd)=component (sourceValues p c.request w m) odd := by
  exact VeriTile.Bench.Repairs.RotaryEmb.Stores.readback _ m (vector_coherent c indices accepted p ws active m) _
    ((vectorWrites_mem c indices p ws m _).mpr ⟨w,hw,odd,rfl⟩)

theorem vector_frame (c : Checked.Call) (indices : Indices) (p : Ops α) (ws : List Work) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w∈ws,¬Writes c.request indices w r a) : vectorBranch c indices p ws m r a=m r a := by
  apply VeriTile.Bench.Repairs.RotaryEmb.Stores.frame
  intro x hx eq
  rcases (vectorWrites_mem c indices p ws m x).mp hx with ⟨w,hw,odd,rfl⟩
  have e := Prod.mk.inj eq
  apply outside w hw
  exact ⟨e.1,by cases odd <;> simp_all⟩

theorem vector_eq_execute (c : Checked.Call) (indices : Indices) (accepted : Checked.check c indices=true) (p : Ops α) (ws : List Work)
    (active : ∀ w∈ws,c.request.Active w) (unique : ws.Nodup) (m : Memory α) :
    vectorBranch c indices p ws m=execute p c.request indices ws m := by
  classical
  funext r a
  by_cases touched : ∃ w∈ws,Writes c.request indices w r a
  · rcases touched with ⟨w,hw,hr,ha|ha⟩
    · subst r; subst a
      rw [vector_readback c indices accepted p ws active m w hw false,
        execute_readback p c indices accepted ws m active unique w hw false]
    · subst r; subst a
      rw [vector_readback c indices accepted p ws active m w hw true,
        execute_readback p c indices accepted ws m active unique w hw true]
  · have outside : ∀ w∈ws,¬Writes c.request indices w r a := by simpa using touched
    rw [vector_frame c indices p ws m r a outside,execute_frame p c.request indices ws m r a outside]



def branchWork (r : Request) (pid : Program) (key : Bool) : List Work :=
  ((((List.range r.blockTokens) ×ˢ (List.range r.blockPairs)).map
    (fun lane => (pid,(key,lane)))).filter (eligible r) |>.map (workOf r)).filter
      (fun w => decide (r.Active w))

def programWork (r : Request) (pid : Program) : List Work :=
  if r.cached then branchWork r pid true ++ branchWork r pid false
  else branchWork r pid false ++ branchWork r pid true

theorem schedule_flat (r : Request) : schedule r=(programs r).flatMap (programWork r) := by
  unfold schedule launches
  simp only [SProd.sprod,List.product,List.map_flatMap,List.filter_flatMap]
  apply congrArg (fun f => (programs r).flatMap f)
  funext pid
  cases cached : r.cached <;>
    simp [lanes,keys,programWork,branchWork,cached,SProd.sprod,List.product,List.map_flatMap,
      List.filter_flatMap,List.map_append,List.filter_append,List.map_map,Function.comp_def]

theorem branch_sub (r : Request) (pid : Program) (hp : pid∈programs r) (key : Bool)
    (w : Work) (hw : w∈branchWork r pid key) : w∈schedule r := by
  rw [schedule_flat]
  apply List.mem_flatMap.mpr
  refine ⟨pid,hp,?_⟩
  cases cached : r.cached <;> cases key <;> simp [programWork,cached,hw]

theorem branch_key (r : Request) (pid : Program) (key : Bool) (w : Work)
    (hw : w∈branchWork r pid key) : w.key=key := by
  rcases List.mem_map.mp (List.mem_filter.mp hw).1 with ⟨launch,hl,rfl⟩
  rcases List.mem_map.mp (List.mem_filter.mp hl).1 with ⟨lane,_,rfl⟩
  rfl

theorem branch_nodup (r : Request) (pid : Program) (hp : pid∈programs r) (key : Bool) :
    (branchWork r pid key).Nodup := by
  have sub : List.Sublist (branchWork r pid key) (schedule r) := by
    rw [schedule_flat]
    have hprog : List.Sublist (programWork r pid) ((programs r).flatMap (programWork r)) :=
      VeriTile.Bench.Repairs.RotaryEmb.list_block_sublist (programWork r) pid (programs r) hp
    cases cached : r.cached <;> cases key
    · exact (List.sublist_append_left _ _).trans (by simpa [programWork,cached] using hprog)
    · exact (List.sublist_append_right _ _).trans (by simpa [programWork,cached] using hprog)
    · exact (List.sublist_append_right _ _).trans (by simpa [programWork,cached] using hprog)
    · exact (List.sublist_append_left _ _).trans (by simpa [programWork,cached] using hprog)
  exact (schedule_nodup r).sublist sub

theorem execute_append (p : Ops α) (r : Request) (indices : Indices) (a b : List Work) (m : Memory α) :
    execute p r indices (a++b) m=execute p r indices b (execute p r indices a m) := by
  induction a generalizing m with
  | nil => rfl
  | cons w ws ih => exact ih (sourceStep p r indices w m)

theorem execute_values_stable (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (ws : List Work) (active : ∀ w∈ws,c.request.Active w) (v : Work) (hv : c.request.Active v)
    (outside : v∉ws) (m : Memory α) :
    sourceValues p c.request v (execute p c.request indices ws m)=sourceValues p c.request v m := by
  induction ws generalizing m with
  | nil => rfl
  | cons w rest ih =>
    simp only [List.mem_cons,not_or] at outside
    change sourceValues p c.request v (execute p c.request indices rest (sourceStep p c.request indices w m))=_
    rw [ih (fun w hw => active w (by simp [hw])) outside.2,
      values_stable p c indices accepted w v (active w (by simp)) hv (Ne.symm outside.1)]

theorem vectorWrites_stable (p : Ops α) (c : Call) (indices : Indices) (accepted : check c indices=true)
    (before after : List Work) (hb : ∀ w∈before,c.request.Active w) (ha : ∀ w∈after,c.request.Active w)
    (apart : ∀ w∈after,w∉before) (m : Memory α) :
    vectorWrites c indices p after (execute p c.request indices before m)=vectorWrites c indices p after m := by
  dsimp only [vectorWrites]
  congr 1 <;> apply List.map_congr_left <;> intro w hw <;>
    rw [execute_values_stable p c indices accepted before hb w (ha w hw) (apart w hw)]

/-- Shared coefficient vectors are loaded once before either Q scatter. -/
def sharedValues (p : Ops α) (r : Request) (w : Work) (data coefficients : Memory α) : α×α :=
  sourcePair p
    (data (.other r.device) (r.data w.key w.token w.head (r.dim w.pair false)))
    (data (.other r.device) (r.data w.key w.token w.head (r.dim w.pair true)))
    (coefficients (.other r.device) (r.coefficient false w.token w.pair))
    (coefficients (.other r.device) (r.coefficient true w.token w.pair))

def sharedWrites (c : Call) (index : Indices) (p : Ops α) (ws : List Work)
    (data coefficients : Memory α) : List (VeriTile.Bench.Repairs.RotaryEmb.Stores.Write α) :=
  let output := fun (odd : Bool) (w : Work) =>
    ((.other c.request.device,destination c.request index w odd),component (sharedValues p c.request w data coefficients) odd)
  ws.map (output false) ++ ws.map (output true)

theorem execute_coefficient (p : Ops α) (c : Call) (index : Indices) (accepted : check c index=true)
    (ws : List Work) (active : ∀ w∈ws,c.request.Active w) (v : Work) (hv : c.request.Active v)
    (m : Memory α) (sine : Bool) :
    execute p c.request index ws m (.other c.request.device) (c.request.coefficient sine v.token v.pair)=
      m (.other c.request.device) (c.request.coefficient sine v.token v.pair) := by
  apply execute_frame
  intro w hw
  rintro ⟨_,same|same⟩
  · exact coefficient_preserved c index accepted w v (active w hw) hv false sine same.symm
  · exact coefficient_preserved c index accepted w v (active w hw) hv true sine same.symm

theorem sharedWrites_eq (p : Ops α) (c : Call) (index : Indices) (accepted : check c index=true)
    (before after : List Work) (hb : ∀ w∈before,c.request.Active w) (ha : ∀ w∈after,c.request.Active w)
    (m : Memory α) : sharedWrites c index p after (execute p c.request index before m) m=
      vectorWrites c index p after (execute p c.request index before m) := by
  dsimp only [sharedWrites,vectorWrites]
  congr 1 <;> apply List.map_congr_left <;> intro w hw <;>
    simp only [sharedValues,sourceValues,
      execute_coefficient p c index accepted before hb w (ha w hw) m false,
      execute_coefficient p c index accepted before hb w (ha w hw) m true]

/-- Cached execution keeps the Q values loaded before either cache scatter.
Uncached execution completes the Q scatters before loading K. -/
def sourceProgram (c : Call) (indices : Indices) (p : Ops α) (pid : Program) (m : Memory α) : Memory α :=
  if c.request.cached then
    let qWrites := vectorWrites c indices p (branchWork c.request pid false) m
    let afterCache := vectorBranch c indices p (branchWork c.request pid true) m
    VeriTile.Bench.Repairs.RotaryEmb.Stores.commit qWrites afterCache
  else
    let afterQ := vectorBranch c indices p (branchWork c.request pid false) m
    VeriTile.Bench.Repairs.RotaryEmb.Stores.commit
      (sharedWrites c indices p (branchWork c.request pid true) afterQ m) afterQ

theorem sourceProgram_eq (c : Call) (indices : Indices) (accepted : check c indices=true) (p : Ops α)
    (pid : Program) (hp : pid∈programs c.request) (m : Memory α) :
    sourceProgram c indices p pid m=execute p c.request indices (programWork c.request pid) m := by
  have active (key : Bool) : ∀ w∈branchWork c.request pid key,c.request.Active w :=
    fun w hw => schedule_active c.request w (branch_sub c.request pid hp key w hw)
  have unique := branch_nodup c.request pid hp
  have apart : ∀ w∈branchWork c.request pid false,w∉branchWork c.request pid true := by
    intro w hq hk
    have q := branch_key c.request pid false w hq
    have k := branch_key c.request pid true w hk
    simp_all
  cases cached : c.request.cached
  · simp only [sourceProgram,programWork,cached,Bool.false_eq_true,↓reduceIte]
    rw [vector_eq_execute c indices accepted p _ (active false) (unique false),
      sharedWrites_eq p c indices accepted _ _ (active false) (active true) m]
    change vectorBranch c indices p _ _=_
    rw [vector_eq_execute c indices accepted p _ (active true) (unique true),execute_append]
  · simp only [sourceProgram,programWork,cached,↓reduceIte]
    rw [vector_eq_execute c indices accepted p _ (active true) (unique true)]
    rw [←vectorWrites_stable p c indices accepted _ _ (active true) (active false) apart m]
    change vectorBranch c indices p _ _=_
    rw [vector_eq_execute c indices accepted p _ (active false) (unique false),execute_append]

def sourcePrograms (c : Call) (indices : Indices) (p : Ops α) : List Program → Memory α → Memory α
  | [],m => m
  | pid::rest,m => sourcePrograms c indices p rest (sourceProgram c indices p pid m)

def source (c : Call) (indices : Indices) (p : Ops α) (m : Memory α) :=
  sourcePrograms c indices p (programs c.request) m

theorem sourcePrograms_eq (c : Call) (indices : Indices) (accepted : check c indices=true) (p : Ops α)
    (ps : List Program) (inside : ∀ pid∈ps,pid∈programs c.request) (m : Memory α) :
    sourcePrograms c indices p ps m=execute p c.request indices (ps.flatMap (programWork c.request)) m := by
  induction ps generalizing m with
  | nil => rfl
  | cons pid rest ih =>
    simp only [sourcePrograms,List.flatMap_cons,execute_append]
    rw [sourceProgram_eq c indices accepted p pid (inside pid (by simp)) m]
    exact ih (fun p hp => inside p (by simp [hp])) _

theorem source_eq_execute (c : Call) (indices : Indices) (accepted : check c indices=true)
    (p : Ops α) (m : Memory α) : source c indices p m=execute p c.request indices (schedule c.request) m := by
  rw [source,sourcePrograms_eq c indices accepted p _ (fun _ h => h),←schedule_flat]

#axiomsClean vector_eq_execute
#axiomsClean sourceProgram_eq
#axiomsClean source_eq_execute
end VeriTile.Bench.Repairs.RotaryEmbNopad
