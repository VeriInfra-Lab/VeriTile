import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Preparation
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Execution

namespace VeriTile.Bench.Repairs.FastLayernorm.Layout
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Execution
noncomputable section
open scoped Classical

def outputs (p : Data) : Fin 3 → Dense
  | ⟨0,_⟩ => p.r | ⟨1,_⟩ => p.mu | ⟨2,_⟩ => p.y

def reads (p : Data) : List Dense := [p.x,p.w]++p.bias.toList

structure Forward (p : Data) (rowCount N : Nat) (callerTensors : List Storage.Tensor) : Prop where
  positive : 0<N
  xCount : p.x.count=rowCount*N
  yCount : p.y.count=rowCount*N
  rCount : p.r.count=rowCount
  muCount : p.mu.count=rowCount
  wCount : N≤p.w.count
  biasCount : ∀ b,p.bias=some b → N≤b.count
  inputsOwned : ∀ input,input∈reads p → input.owner=p.x.owner
  outputsOwned : ∀ k,(outputs p k).owner=p.x.owner
  separated : ∀ k j,k≠j → Storage.Disjoint (outputs p k).tensor.backing (outputs p j).tensor.backing
  freshInputs : ∀ k input,input∈reads p → Storage.Disjoint (outputs p k).tensor.backing input.tensor.backing
  callerFresh : ∀ k old,old∈callerTensors → Storage.Disjoint (outputs p k).tensor.backing old.backing

def forwardTrace (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (h : Forward p rowCount N callerTensors) (eps : α) (m : Storage.World Store) : Trace α 3 where
  owner := p.x.owner
  regions := match p.bias with
    | some bias => forwardRegions ops plan api p.x p.w bias p.y p.r p.mu eps m
    | none => forwardRegionsNoBias ops plan api p.x p.w p.y p.r p.mu eps m
  owned := by
    intro k
    have ho := h.outputsOwned k
    cases hb : p.bias <;> fin_cases k <;>
      simpa [forwardRegions,forwardRegionsNoBias,outputs] using ho
  separated := by
    intro k j hkj
    have hs := h.separated k j hkj
    cases hb : p.bias <;> fin_cases k <;> fin_cases j <;>
      simpa [forwardRegions,forwardRegionsNoBias,outputs] using hs

variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {lib : APIHost api neg inputs Metadata} {x w : Storage.Tensor} {bias : Option Storage.Tensor}
  {rowCount N : Nat} {initial : State Store Metadata}

theorem route_present (r : Route lib x w bias rowCount N initial) :
    (∀ input,input∈reads r.data → r.final.live input.tensor) ∧
    (∀ k,r.final.live (outputs r.data k).tensor) := by
  have hx := (materialize_metadata r.xr).2.2.2
  have hw := (materialize_metadata r.wr).2.2.2
  have hy := (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).1.present
  have hr := (lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK).1.present
  have hm := (lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK).1.present
  constructor
  · intro input hi
    simp only [reads,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hi
    rcases hi with (hx'|hw')|hb
    · subst input
      exact r.tailX.live _ hx
    · subst input
      exact r.tailWeight.live _ hw
    · cases hbr : r.br with
      | none => simp [Route.data,hbr,OptionalRoute.output] at hb
      | some br =>
        have he : input=br.output := by simpa [Route.data,hbr,OptionalRoute.output] using hb
        subst input
        apply r.tailBias.live
        simpa [hbr,OptionalRoute.final] using (materialize_metadata br).2.2.2
  · intro k
    fin_cases k
    · exact (lib.emptyStats x.view.device rowCount).transition r.sR |>.live _ hr
    · exact hm
    · exact (lib.emptyStats x.view.device rowCount).transition r.sR |>.live _
        ((lib.emptyStats x.view.device rowCount).transition r.sY |>.live _ hy)

theorem route_counts (r : Route lib x w bias rowCount N initial)
    (shape : x.view.shape.prod=rowCount*N) :
    r.data.x.count=rowCount*N ∧ r.data.y.count=rowCount*N ∧
    r.data.r.count=rowCount ∧ r.data.mu.count=rowCount ∧ r.data.w.count=w.view.shape.prod := by
  have hx := materialize_metadata r.xr
  have hw := materialize_metadata r.wr
  have hy := lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK
  have hr := lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK
  have hm := lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK
  refine ⟨hx.2.1.trans shape,?_,?_,?_,hw.2.1⟩
  · change r.y.count=_
    rw [r.y.countEq,hy.2.1,←r.xr.output.countEq,hx.2.1,shape]
  · change r.r.count=_
    simp [r.r.countEq,hr.2.1]
  · change r.mu.count=_
    simp [r.mu.countEq,hm.2.1]

/-- All allocation freshness facts below come from the actual empty_like and
empty calls recorded by the preparation route, including their order. -/
theorem route_output_owners (r : Route lib x w bias rowCount N initial) :
    ∀ k,(outputs r.data k).owner=r.data.x.owner := by
  have hx := (materialize_metadata r.xr).2.2.1
  have hy := (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).2.2
  have hr := (lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK).2.2
  have hm := (lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK).2.2
  intro k
  fin_cases k
  · exact hr.trans hx.symm
  · exact hm.trans hx.symm
  · exact hy

theorem route_caller_fresh (r : Route lib x w bias rowCount N initial) :
    ∀ k old,old∈inputs → Storage.Disjoint (outputs r.data k).tensor.backing old.backing := by
  intro k old ho
  fin_cases k
  · exact (lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK).1.callerFresh old ho
  · exact (lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK).1.callerFresh old ho
  · exact (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).1.callerFresh old ho

theorem route_reads_before (r : Route lib x w bias rowCount N initial) :
    ∀ input,input∈reads r.data → r.br.final.live input.tensor := by
  intro input hi
  simp only [reads,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hi
  rcases hi with (hx|hw)|hb
  · subst input
    exact r.br.transition.live _ (r.wr.transition.live _ (materialize_metadata r.xr).2.2.2)
  · subst input
    exact r.br.transition.live _ (materialize_metadata r.wr).2.2.2
  · cases hbr : r.br with
    | none => simp [Route.data,hbr,OptionalRoute.output] at hb
    | some br =>
      have he : input=br.output := by simpa [Route.data,hbr,OptionalRoute.output] using hb
      subst input
      simpa [hbr,OptionalRoute.final] using (materialize_metadata br).2.2.2

theorem route_input_owners (r : Route lib x w bias rowCount N initial) :
    ∀ input,input∈reads r.data → input.owner=r.data.x.owner := by
  intro input hi
  simp only [reads,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at hi
  rcases hi with (hx|hw)|hb
  · subst input; rfl
  · subst input
    exact (materialize_metadata r.wr).2.2.1.trans (materialize_metadata r.xr).2.2.1.symm
  · cases hbr : r.br with
    | none => simp [Route.data,hbr,OptionalRoute.output] at hb
    | some br =>
      have he : input=br.output := by simpa [Route.data,hbr,OptionalRoute.output] using hb
      subst input
      exact (materialize_metadata br).2.2.1.trans (materialize_metadata r.xr).2.2.1.symm

theorem route_fresh_inputs (r : Route lib x w bias rowCount N initial) :
    ∀ k input,input∈reads r.data → Storage.Disjoint (outputs r.data k).tensor.backing input.tensor.backing := by
  intro k input hi
  have live := route_reads_before r input hi
  fin_cases k
  · exact (lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK).1.fresh _
      ((lib.emptyLike r.xr.output).transition r.br.final |>.live _ live)
  · exact (lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK).1.fresh _
      ((lib.emptyStats x.view.device rowCount).transition r.sY |>.live _
        ((lib.emptyLike r.xr.output).transition r.br.final |>.live _ live))
  · exact (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).1.fresh _ live

theorem route_separated (r : Route lib x w bias rowCount N initial) :
    ∀ k j,k≠j → Storage.Disjoint (outputs r.data k).tensor.backing (outputs r.data j).tensor.backing := by
  have hy := (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).1.present
  have hr := (lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK).1.present
  have ry := (lib.emptyStatsResult x.view.device rowCount r.sY r.r r.rOK).1.fresh _ hy
  have my := (lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK).1.fresh _
    ((lib.emptyStats x.view.device rowCount).transition r.sY |>.live _ hy)
  have mr := (lib.emptyStatsResult x.view.device rowCount r.sR r.mu r.muOK).1.fresh _ hr
  intro k j h
  fin_cases k <;> fin_cases j
  · exact False.elim (h rfl)
  · exact disjoint_symm _ _ mr
  · exact ry
  · exact mr
  · exact False.elim (h rfl)
  · exact my
  · exact disjoint_symm _ _ ry
  · exact disjoint_symm _ _ my
  · exact False.elim (h rfl)

theorem route_layout (r : Route lib x w bias rowCount N initial)
    (positive : 0<N) (shape : x.view.shape.prod=rowCount*N)
    (weight : N≤w.view.shape.prod) (biasCount : ∀ b,bias=some b → N≤b.view.shape.prod) :
    Forward r.data rowCount N inputs := by
  obtain ⟨hx,hy,hr,hm,hw⟩ := route_counts r shape
  refine ⟨positive,hx,hy,hr,hm,by omega,?_,route_input_owners r,route_output_owners r,
    route_separated r,route_fresh_inputs r,route_caller_fresh r⟩
  intro b hb
  cases hbr : r.br with
  | none => simp [Route.data,hbr,OptionalRoute.output] at hb
  | @some original br =>
    have he : b=br.output := by simpa [Route.data,hbr,OptionalRoute.output] using hb.symm
    subst b
    rw [(materialize_metadata br).2.1]
    exact biasCount original rfl

#axiomsClean route_layout
#axiomsClean route_present
#axiomsClean route_counts
#axiomsClean route_output_owners
#axiomsClean route_caller_fresh
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Layout
