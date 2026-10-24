import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RawBackendIO

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
open VeriTile.Triton FP.Structural FP.ProfiledRewrite Math Program Backend
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def InputEq {α : Type} (r : Regions) (s a : State α) : Prop :=
  (∀ off, s.mem r.dt off = a.mem r.dt off) ∧
  (∀ off, s.mem r.bias off = a.mem r.bias off) ∧
  (∀ off, s.mem r.rate off = a.mem r.rate off)

theorem logical_congr {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (s a : State α) (same : InputEq r s a) :
    logicalPrepared c p r hasBias useSoftplus lower upper s =
      logicalPrepared c p r hasBias useSoftplus lower upper a ∧
    logicalCumulative c p r hasBias useSoftplus lower upper s =
      logicalCumulative c p r hasBias useSoftplus lower upper a := by
  have hp : logicalPrepared c p r hasBias useSoftplus lower upper s =
      logicalPrepared c p r hasBias useSoftplus lower upper a := by
    funext b h k j
    simp only [logicalPrepared,same.1,same.2.1]
  refine ⟨hp,?_⟩
  funext b h k j
  simp only [logicalCumulative,hp,logicalRate,same.2.2]

def Values {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (hasBias useSoftplus : Bool) (lower upper : α)
    (a : State α) (ls : List Launch) (v : State α) : Prop :=
  ∀ l ∈ ls, ∀ i : TileIndex [c.BH,c.BC], activeOutput c.layout l i.1.val i.2.1.val →
    v.mem r.out (outAddress c.layout l i.1.val i.2.1.val) =
      .mk .real (logicalPrepared c p r hasBias useSoftplus lower upper a
        l.batch (head c.layout l i.1.val) l.chunk i.2.1.val) ∧
    v.mem r.sum (outAddress c.layout l i.1.val i.2.1.val) =
      .mk .real (logicalCumulative c p r hasBias useSoftplus lower upper a
        l.batch (head c.layout l i.1.val) l.chunk i.2.1)

def Frame {α : Type} (c : Whole.Host) (r : Regions) (ls : List Launch) (s v : State α) : Prop :=
  ∀ rg off, ¬ Whole.footprint c r ls (rg,off) → v.mem rg off = s.mem rg off

theorem reads_preserved {α : Type} (c : Whole.Host) (r : Regions) (fresh : Whole.Fresh r)
    (ls : List Launch) (s v a : State α) (same : InputEq r s a) (frame : Frame c r ls s v) :
    InputEq r v a := by
  have read (rg : RegionName) (ho : rg ≠ r.out) (hs : rg ≠ r.sum) (off : Nat) :
      v.mem rg off = s.mem rg off := by
    apply frame rg off
    rintro ⟨l,_,h⟩
    exact Whole.outside_other c r l rg ho hs off h
  exact ⟨fun off => (read _ fresh.dtOut fresh.dtSum off).trans (same.1 off),
    fun off => (read _ fresh.biasOut fresh.biasSum off).trans (same.2.1 off),
    fun off => (read _ fresh.rateOut fresh.rateSum off).trans (same.2.2 off)⟩

theorem tiles_correct {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (fresh : Whole.Fresh r) (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (valid : ∀ l ∈ ls, c.valid l) (s a : State α) (same : InputEq r s a) :
    ∃ v, RawSource.runTiles p.model p.minimum c (RawSource.source c r hasBias useSoftplus) lower upper ls s = some v ∧
      Values c p r hasBias useSoftplus lower upper a ls v ∧ Frame c r ls s v := by
  induction ls generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro l hl; simp at hl
    · intro _ _ _; rfl
  | cons l rest ih =>
    have hl := valid l List.mem_cons_self
    have hrest : ∀ q ∈ rest, c.valid q := fun q hq => valid q (List.mem_cons_of_mem l hq)
    let middle := (sourceRun c p r hasBias useSoftplus lower upper l s).val
    have mf : Frame c r [l] s middle := by
      intro rg off outside
      apply tile_frame c p r hasBias useSoftplus lower upper l s rg off
      exact fun h => outside ⟨l,by simp,h⟩
    have mv : Values c p r hasBias useSoftplus lower upper a [l] middle := by
      intro q hq i active
      have he : q = l := by simpa using hq
      subst q
      have h := tile_values c p r fresh hasBias useSoftplus lower upper l hl s i active
      have hc := logical_congr c p r hasBias useSoftplus lower upper s a same
      simpa only [hc.1,hc.2] using h
    obtain ⟨v,ran,values,frame⟩ := ih hrest middle (reads_preserved c r fresh [l] s middle a same mf)
    refine ⟨v,?_,?_,?_⟩
    · simp only [RawSource.runTiles]
      rw [(sourceRun c p r hasBias useSoftplus lower upper l s).property,Option.bind_some]
      exact ran
    · intro q hq i active
      by_cases hr : q ∈ rest
      · exact values q hr i active
      · have he : q = l := (List.mem_cons.mp hq).resolve_right hr
        subst q
        have ho : ¬ Whole.footprint c r rest (r.out,outAddress c.layout l i.1.val i.2.1.val) := by
          rintro ⟨q,hq,hw⟩
          exact (Whole.other_tile_outside c r fresh l q hl (hrest q hq)
            (fun eq => hr (eq ▸ hq)) i.1 i.2.1 active).1 hw
        have hs : ¬ Whole.footprint c r rest (r.sum,outAddress c.layout l i.1.val i.2.1.val) := by
          rintro ⟨q,hq,hw⟩
          exact (Whole.other_tile_outside c r fresh l q hl (hrest q hq)
            (fun eq => hr (eq ▸ hq)) i.1 i.2.1 active).2 hw
        rw [frame _ _ ho,frame _ _ hs]
        exact mv l (by simp) i active
    · intro rg off outside
      have h1 : ¬ Whole.footprint c r [l] (rg,off) := by
        rintro ⟨q,hq,hw⟩
        have he : q = l := by simpa using hq
        subst q
        exact outside ⟨l,List.mem_cons_self,hw⟩
      have h2 : ¬ Whole.footprint c r rest (rg,off) := by
        rintro ⟨q,hq,hw⟩
        exact outside ⟨q,List.mem_cons_of_mem l hq,hw⟩
      exact (frame rg off h2).trans (mf rg off h1)

theorem grid_correct {α : Type} [Inhabited α] (c : Whole.Host) (p : Primitives α c)
    (r : Regions) (fresh : Whole.Fresh r) (hasBias useSoftplus : Bool) (lower upper : α)
    (ls : List Launch) (schedule : ls.Perm c.grid) (s : State α) :
    ∃ v, RawSource.runTiles p.model p.minimum c (RawSource.source c r hasBias useSoftplus) lower upper ls s = some v ∧
      Values c p r hasBias useSoftplus lower upper s c.grid v ∧ Frame c r c.grid s v := by
  obtain ⟨v,ran,values,frame⟩ := tiles_correct c p r fresh hasBias useSoftplus lower upper ls
    (fun l hl => (c.mem_grid l).1 (schedule.mem_iff.mp hl)) s s
    ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl⟩
  refine ⟨v,ran,?_,?_⟩
  · intro l hl
    exact values l (schedule.mem_iff.mpr hl)
  · intro rg off outside
    apply frame rg off
    rintro ⟨l,hl,hw⟩
    exact outside ⟨l,schedule.mem_iff.mp hl,hw⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.RawBackend
