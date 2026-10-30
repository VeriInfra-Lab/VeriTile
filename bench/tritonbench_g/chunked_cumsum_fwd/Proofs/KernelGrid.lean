import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelIO

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
open VeriTile.Triton FP.Structural
open Math
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

def footprint (c : Geometry) (ls : List Launch) (cell : RegionName × Nat) : Prop :=
  ∃ l ∈ ls, tileFootprint c l cell

theorem outside_other (c : Geometry) (l : Launch) (rg : RegionName)
    (ho : rg ≠ "out") (hs : rg ≠ "sum") (off : Nat) :
    ¬ tileFootprint c l (rg,off) := by
  rintro (h | h)
  · exact ho h.1
  · exact hs h.1

theorem other_tile_outside (c : Geometry) (l q : Launch)
    (hl : c.valid l) (hq : c.valid q) (hne : l ≠ q)
    (i : Fin c.BH) (j : Fin c.BC) (hi : activeOutput c.layout l i.val j.val) :
    ¬ tileFootprint c q ("out",outAddress c.layout l i.val j.val) ∧
    ¬ tileFootprint c q ("sum",outAddress c.layout l i.val j.val) := by
  constructor
  · rintro (⟨_,i',j',ha,he⟩ | ⟨he,_⟩)
    · exact hne (output_unique c l q hl hq i i' j j' hi ha he).1
    · exact (by decide : ("out" : RegionName) ≠ "sum") he
  · rintro (⟨he,_⟩ | ⟨_,i',j',ha,he⟩)
    · exact (by decide : ("sum" : RegionName) ≠ "out") he
    · exact hne (output_unique c l q hl hq i i' j j' hi ha he).1

def InputEq {α : Type} (s a : State α) : Prop :=
  (∀ off, s.mem "dt" off = a.mem "dt" off) ∧
  (∀ off, s.mem "bias" off = a.mem "bias" off) ∧
  (∀ off, s.mem "rate" off = a.mem "rate" off)

theorem logical_congr {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (s a : State α) (same : InputEq s a) :
    logicalPrepared c p hasBias useSoftplus lower upper s =
      logicalPrepared c p hasBias useSoftplus lower upper a ∧
    logicalCumulative c p hasBias useSoftplus lower upper s =
      logicalCumulative c p hasBias useSoftplus lower upper a := by
  have hp : logicalPrepared c p hasBias useSoftplus lower upper s =
      logicalPrepared c p hasBias useSoftplus lower upper a := by
    funext b h k j
    simp only [logicalPrepared,same.1,same.2.1]
  refine ⟨hp,?_⟩
  funext b h k j
  simp only [logicalCumulative,hp,logicalRate,same.2.2]

def Values {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (a : State α) (ls : List Launch) (v : State α) : Prop :=
  ∀ l ∈ ls, ∀ i : TileIndex [c.BH,c.BC], activeOutput c.layout l i.1.val i.2.1.val →
    v.mem "out" (outAddress c.layout l i.1.val i.2.1.val) =
      .mk .real (logicalPrepared c p hasBias useSoftplus lower upper a
        l.batch (head c.layout l i.1.val) l.chunk i.2.1.val) ∧
    v.mem "sum" (outAddress c.layout l i.1.val i.2.1.val) =
      .mk .real (logicalCumulative c p hasBias useSoftplus lower upper a
        l.batch (head c.layout l i.1.val) l.chunk i.2.1)

def Frame {α : Type} (c : Geometry) (ls : List Launch) (s v : State α) : Prop :=
  ∀ rg off, ¬ footprint c ls (rg,off) → v.mem rg off = s.mem rg off

theorem reads_preserved {α : Type} (c : Geometry)
    (ls : List Launch) (s v a : State α) (same : InputEq s a) (frame : Frame c ls s v) :
    InputEq v a := by
  have read (rg : RegionName) (ho : rg ≠ "out") (hs : rg ≠ "sum") (off : Nat) :
      v.mem rg off = s.mem rg off := by
    apply frame rg off
    rintro ⟨l,_,h⟩
    exact outside_other c l rg ho hs off h
  exact ⟨fun off => (read _ (by decide) (by decide) off).trans (same.1 off),
    fun off => (read _ (by decide) (by decide) off).trans (same.2.1 off),
    fun off => (read _ (by decide) (by decide) off).trans (same.2.2 off)⟩

theorem tiles_correct {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (sizes : Sizes c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (valid : ∀ l ∈ ls, c.valid l) (s a : State α) (same : InputEq s a) :
    ∃ v, runTiles p.model c hasBias useSoftplus lower upper ls s = some v ∧
      Values c p hasBias useSoftplus lower upper a ls v ∧ Frame c ls s v := by
  induction ls generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro l hl; simp at hl
    · intro _ _ _; rfl
  | cons l rest ih =>
    have hl := valid l List.mem_cons_self
    have hrest : ∀ q ∈ rest, c.valid q := fun q hq => valid q (List.mem_cons_of_mem l hq)
    let middle := (sourceRun c p hasBias useSoftplus lower upper l s).val
    have mf : Frame c [l] s middle := by
      intro rg off outside
      apply tile_frame c p sizes hasBias useSoftplus lower upper l hl s rg off
      exact fun h => outside ⟨l,by simp,h⟩
    have mv : Values c p hasBias useSoftplus lower upper a [l] middle := by
      intro q hq i active
      have he : q = l := by simpa using hq
      subst q
      have h := tile_values c p sizes hasBias useSoftplus lower upper l hl s i active
      have hc := logical_congr c p hasBias useSoftplus lower upper s a same
      simpa only [hc.1,hc.2] using h
    obtain ⟨v,ran,values,frame⟩ := ih hrest middle (reads_preserved c [l] s middle a same mf)
    refine ⟨v,?_,?_,?_⟩
    · simp only [runTiles]
      rw [(sourceRun c p hasBias useSoftplus lower upper l s).property]
      exact ran
    · intro q hq i active
      by_cases hr : q ∈ rest
      · exact values q hr i active
      · have he : q = l := (List.mem_cons.mp hq).resolve_right hr
        subst q
        have ho : ¬ footprint c rest ("out",outAddress c.layout l i.1.val i.2.1.val) := by
          rintro ⟨q,hq,hw⟩
          exact (other_tile_outside c l q hl (hrest q hq)
            (fun eq => hr (eq ▸ hq)) i.1 i.2.1 active).1 hw
        have hs : ¬ footprint c rest ("sum",outAddress c.layout l i.1.val i.2.1.val) := by
          rintro ⟨q,hq,hw⟩
          exact (other_tile_outside c l q hl (hrest q hq)
            (fun eq => hr (eq ▸ hq)) i.1 i.2.1 active).2 hw
        rw [frame _ _ ho,frame _ _ hs]
        exact mv l (by simp) i active
    · intro rg off outside
      have h1 : ¬ footprint c [l] (rg,off) := by
        rintro ⟨q,hq,hw⟩
        have he : q = l := by simpa using hq
        subst q
        exact outside ⟨l,List.mem_cons_self,hw⟩
      have h2 : ¬ footprint c rest (rg,off) := by
        rintro ⟨q,hq,hw⟩
        exact outside ⟨q,List.mem_cons_of_mem l hq,hw⟩
      exact (frame rg off h2).trans (mf rg off h1)

theorem grid_correct {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (sizes : Sizes c)
    (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (schedule : ls.Perm c.grid) (s : State α) :
    ∃ v, runTiles p.model c hasBias useSoftplus lower upper ls s = some v ∧
      Values c p hasBias useSoftplus lower upper s c.grid v ∧ Frame c c.grid s v := by
  obtain ⟨v,ran,values,frame⟩ := tiles_correct c p sizes hasBias useSoftplus lower upper ls
    (fun l hl => (c.mem_grid l).1 (schedule.mem_iff.mp hl)) s s
    ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl⟩
  refine ⟨v,ran,?_,?_⟩
  · intro l hl
    exact values l (schedule.mem_iff.mpr hl)
  · intro rg off outside
    apply frame rg off
    rintro ⟨l,hl,hw⟩
    exact outside ⟨l,schedule.mem_iff.mp hl,hw⟩


/-- Complete logical output coverage, including chunk padding and empty grids. -/
theorem complete {α : Type} [Inhabited α] (c : Geometry) (p : Primitives α c)
    (sizes : Sizes c) (hasBias useSoftplus : Bool) (lower upper : α) (s : State α) :
    ∃ v, launch p.model c hasBias useSoftplus lower upper s = some v ∧
      (∀ b h k j, b<c.batches → h<c.heads → k<c.chunks → ∀ hj : j<c.chunk,
        v.mem "out" (packed c b h k j) =
          .mk .real (logicalPrepared c p hasBias useSoftplus lower upper s b h k j) ∧
        v.mem "sum" (packed c b h k j) =
          .mk .real (logicalCumulative c p hasBias useSoftplus lower upper s b h k
            ⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩)) ∧ Frame c c.grid s v := by
  obtain ⟨v,ran,values,frame⟩ := grid_correct c p sizes hasBias useSoftplus lower upper
    c.grid (List.Perm.refl _) s
  refine ⟨v,ran,?_,frame⟩
  intro b h k j hb hh hk hj
  obtain ⟨l,hl,i,t,active,hb',hh',hk',hj'⟩ := cover c b h k j hb hh hk hj
  have out := values l hl (i,t,PUnit.unit) active
  have index : t = (⟨j,lt_of_lt_of_le hj c.chunk_le_BC⟩ : Fin c.BC) := Fin.ext hj'
  rw [out_packed,hb',hh',hk',hj',index] at out
  exact out

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel
