import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
open VeriTile.Triton Math Program Initialization
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 10000

/-- `torch.empty` creates two fresh outputs. Existing input aliases are allowed. -/
structure Fresh (r : Regions) : Prop where
  dtOut : r.dt ≠ r.out
  dtSum : r.dt ≠ r.sum
  rateOut : r.rate ≠ r.out
  rateSum : r.rate ≠ r.sum
  biasOut : r.bias ≠ r.out
  biasSum : r.bias ≠ r.sum
  outSum : r.out ≠ r.sum

def source32 (c : Host) (r : Regions) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) :=
  (kernel c.layout r .fp32 Bool.false .fp32 .fp32 hasBias useSoftplus lower upper).eraseDType.toAlgKernel

def runTiles (k : Kernel) : List Launch → BlockState → Option BlockState
  | [],s => some s
  | l::rest,s => (exec k (tileState l s)).bind (runTiles k rest)

def footprint (c : Host) (r : Regions) (ls : List Launch) : WriteFootprint :=
  fun cell => ∃ l ∈ ls, Execution.footprint c.layout l r cell

def Reads (r : Regions) (s : BlockState) (a : Inputs) : Prop :=
  (∀ off, s.readMem r.dt off = a.dt off) ∧
  (∀ off, s.readMem r.bias off = a.bias off) ∧
  (∀ off, s.readMem r.rate off = a.rate off)

def Values (c : Host) (r : Regions) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (ls : List Launch) (v : BlockState) : Prop :=
  ∀ l ∈ ls, ∀ (i : Fin c.layout.BH) (j : Fin c.layout.BC),
    activeOutput c.layout l i.val j.val →
    v.readMem r.out (outAddress c.layout l i.val j.val) =
      prepared c.layout l a hasBias useSoftplus lower upper i.val j.val ∧
    v.readMem r.sum (sumAddress c.layout l i.val j.val) =
      cumulative c.layout l a hasBias useSoftplus lower upper i j

theorem outside_other (c : Host) (r : Regions) (l : Launch) (rg : RegionName)
    (ho : rg ≠ r.out) (hs : rg ≠ r.sum) (off : Nat) :
    ¬ Execution.footprint c.layout l r (rg,off) := by
  rintro (h | h)
  · exact ho h.1
  · exact hs h.1

theorem reads_preserved (c : Host) (r : Regions) (fresh : Fresh r) (l : Launch)
    (u v : BlockState) (a : Inputs) (hr : Reads r u a)
    (hf : BlockState.WriteWithin (Execution.footprint c.layout l r) u v) : Reads r v a := by
  have same (rg : RegionName) (ho : rg ≠ r.out) (hs : rg ≠ r.sum) (off : Nat) :
      v.readMem rg off = u.readMem rg off := by
    unfold BlockState.readMem
    rw [← hf rg off (outside_other c r l rg ho hs off)]
  exact ⟨fun off => (same _ fresh.dtOut fresh.dtSum off).trans (hr.1 off),
    fun off => (same _ fresh.biasOut fresh.biasSum off).trans (hr.2.1 off),
    fun off => (same _ fresh.rateOut fresh.rateSum off).trans (hr.2.2 off)⟩

theorem other_tile_outside (c : Host) (r : Regions) (fresh : Fresh r) (l q : Launch)
    (hl : c.valid l) (hq : c.valid q) (hne : l ≠ q)
    (i : Fin c.layout.BH) (j : Fin c.layout.BC)
    (hi : activeOutput c.layout l i.val j.val) :
    ¬ Execution.footprint c.layout q r (r.out,outAddress c.layout l i.val j.val) ∧
    ¬ Execution.footprint c.layout q r (r.sum,sumAddress c.layout l i.val j.val) := by
  constructor
  · rintro (⟨_,i',j',ha,he⟩ | ⟨he,_⟩)
    · exact hne (output_unique c l q hl hq i i' j j' hi ha he).1
    · exact fresh.outSum he
  · rintro (⟨he,_⟩ | ⟨_,i',j',ha,he⟩)
    · exact fresh.outSum he.symm
    · exact hne (output_unique c l q hl hq i i' j j' hi ha he).1

/-- Every list element runs the complete original source. The result holds for
all scheduling orders, with repeated programs also harmless. -/
theorem tiles_correct (c : Host) (r : Regions) (fresh : Fresh r)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (ls : List Launch) (valid : ∀ l ∈ ls, c.valid l)
    (s : BlockState) (a : Inputs) (read : Reads r s a) :
    ∃ v, runTiles (source32 c r hasBias useSoftplus lower upper) ls s = some v ∧
      Values c r a hasBias useSoftplus lower upper ls v ∧
      BlockState.WriteWithin (footprint c r ls) s v := by
  induction ls generalizing s with
  | nil =>
    refine ⟨s,rfl,?_,?_⟩
    · intro l h; simp at h
    · intro rg off _; rfl
  | cons l rest ih =>
    have hl := valid l List.mem_cons_self
    have hrest : ∀ q ∈ rest, c.valid q := fun q hq => valid q (List.mem_cons_of_mem l hq)
    obtain ⟨middle,ran,values,frame⟩ := Execution.correct c.layout r .fp32 Bool.false .fp32 .fp32 a
      hasBias useSoftplus lower upper (tileState l s) read.1 read.2.1 read.2.2 fresh.rateOut fresh.outSum
      (by simpa using noCollision c l hl) (by simpa using noCollision c l hl)
    have hv : Values c r a hasBias useSoftplus lower upper [l] middle := by
      intro q hq
      have he : q = l := by simpa using hq
      subst q
      simpa using values
    have hf : BlockState.WriteWithin (Execution.footprint c.layout l r) s middle := by
      simpa only [launch_tileState] using frame
    obtain ⟨v,tail,out,fr⟩ := ih hrest middle (reads_preserved c r fresh l s middle a read hf)
    refine ⟨v,?_,?_,?_⟩
    · simp only [runTiles,source32] at ran ⊢
      rw [ran,Option.bind_some]
      exact tail
    · intro q hq i j active
      by_cases hr : q ∈ rest
      · exact out q hr i j active
      · have he : q = l := (List.mem_cons.mp hq).resolve_right hr
        subst q
        have ho : ¬ footprint c r rest (r.out,outAddress c.layout l i.val j.val) := by
          rintro ⟨q,hq,hwrite⟩
          exact (other_tile_outside c r fresh l q hl (hrest q hq)
            (fun eq => hr (eq ▸ hq)) i j active).1 hwrite
        have hs : ¬ footprint c r rest (r.sum,sumAddress c.layout l i.val j.val) := by
          rintro ⟨q,hq,hwrite⟩
          exact (other_tile_outside c r fresh l q hl (hrest q hq)
            (fun eq => hr (eq ▸ hq)) i j active).2 hwrite
        have vo := hv l (by simp) i j active
        simpa only [BlockState.readMem,← fr _ _ ho,← fr _ _ hs] using vo
    · intro rg off hn
      have h1 : ¬ Execution.footprint c.layout l r (rg,off) :=
        fun h => hn ⟨l,List.mem_cons_self,h⟩
      have h2 : ¬ footprint c r rest (rg,off) := by
        rintro ⟨q,hq,h⟩
        exact hn ⟨q,List.mem_cons_of_mem l hq,h⟩
      exact (hf rg off h1).trans (fr rg off h2)

theorem grid_correct (c : Host) (r : Regions) (fresh : Fresh r)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (ls : List Launch) (schedule : ls.Perm c.grid) (s : BlockState) :
    ∃ v, runTiles (source32 c r hasBias useSoftplus lower upper) ls s = some v ∧
      Values c r ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩
        hasBias useSoftplus lower upper c.grid v ∧
      BlockState.WriteWithin (footprint c r c.grid) s v := by
  obtain ⟨v,he,ho,hf⟩ := tiles_correct c r fresh hasBias useSoftplus lower upper ls
    (fun l hl => (c.mem_grid l).1 (schedule.mem_iff.mp hl)) s
    ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩ ⟨fun _ => rfl,fun _ => rfl,fun _ => rfl⟩
  refine ⟨v,he,?_,?_⟩
  · intro l hl; exact ho l (schedule.mem_iff.mpr hl)
  · intro rg off hn
    apply hf rg off
    rintro ⟨l,hl,hw⟩
    exact hn ⟨l,schedule.mem_iff.mp hl,hw⟩

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
