import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordGrid
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.TotalScaled
open VeriTile.Triton FP.Structural FP.IntegerConversion
open ScaledWord GridExecution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Total source conversion: undefined software casts retain the actual local
conversion port instead of excluding an admitted scale or input value. -/
def expected (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) (i : CIndex p) :=
  totalSigned (u 32) 32 (scaled cfg p s i)
def logicalResult (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word) (r c : Nat) :=
  totalSigned (u 32) 32 (logicalScaled cfg p s r c)

def written (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p initial i then
      t.write p.output (outputOffset p initial i) (.mk .int (expected cfg u p initial i)) else t) initial

attribute [local irreducible] FP.mul FP.round toSigned totalSigned

theorem epilogueMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial current : State Word)
    (h : Invariant p initial p.steps current) :
    (epilogueExecution rest cfg u p initial current h).val.mem = (written cfg u p initial).mem := by
  simp +contextual [epilogueExecution, written, masked_scatter_memory, h.memory,
    active, outputOffset, outputRow, outputColumn, expected, scaled,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    and_assoc, and_comm, and_left_comm]

theorem executionMemory (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial = some final ∧
      final.mem = (written cfg u p initial).mem := by
  let start := preludeExecution rest cfg u p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution rest cfg u p initial start.val 0 (by omega)
    (preludeInvariant rest cfg u p initial)
  let finish := epilogueExecution rest cfg u p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory rest cfg u p initial mid hi⟩
  change run (backend rest cfg u) (source p).surfaceBody initial = some finish.val
  rw [show (source p).surfaceBody = prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p from rfl,
    run_append, run_append, start.property]
  have hsingle (s : State Word) :
      run (backend rest cfg u) [.forRange "j" 0 p.steps 1 (body p)] s =
        FP.Structural.range (backend rest cfg u) "j" 0 p.steps 1 (body p) s := by simp [run,step]
  simp only [Option.bind_some, hsingle, hm]
  exact finish.property

theorem correctness (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (p : Parameters) (initial : State Word)
    (hw : ∀ i j, active p initial i → active p initial j →
      outputOffset p initial i = outputOffset p initial j → i = j) :
    ∃ final, FP.Structural.exec (backend rest cfg u) (source p) initial = some final ∧
      Post p initial final (expected cfg u p initial) := by
  obtain ⟨final,he,hm⟩ := executionMemory rest cfg u p initial
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o hframe
    rw [hm]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (hframe.imp id (fun h i _ => h i)) _

theorem expected_logical (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (s : State Word)
    (he : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hc : p.inner≤p.steps*p.blockInner)
    (i : CIndex p) (hi : active p s i) :
    expected cfg u p s i=logicalResult cfg u p s (outputRow p s i) (outputColumn p s i) := by
  unfold expected logicalResult
  rw [scaled_logical cfg p s he hc i hi]

theorem result_preserved (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters) (initial current : State Word)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output)
    (hout : Outside p.output initial current) (r c : Nat) :
    logicalResult cfg u p current r c=logicalResult cfg u p initial r c := by
  unfold logicalResult
  rw [scaled_preserved cfg p initial current ha hb hs hout]

theorem single_logical (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p))
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hscale : p.scale≠p.output)
    (initial : State Word) (current : State Word) (hout : Outside p.output initial current) (pid : Nat) :
    ∃ final, one rest cfg u p pid current=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalResult cfg u p initial x.1.val x.2.val))
        (Geometry.Owns (geometry p) pid) current final := by
  let s := atProgram current pid
  have hw : ∀ i j, active p s i → active p s j → outputOffset p s i=outputOffset p s j → i=j := by
    intro i j hi hj he
    apply Geometry.offset_injective (geometry p) hgeo pid i j hi hj
    simpa only [s,offset_at] using he
  obtain ⟨final,he,hread,hframe⟩ := correctness rest cfg u p s hw
  refine ⟨final,he,?_,?_⟩
  · intro x hx
    obtain ⟨i,hr,hc⟩ := hx
    have hi : active p s i := Geometry.owns_active (geometry p) pid x i hr hc
    have ho : outputOffset p s i=Geometry.address (geometry p) x := by
      rw [show s=atProgram current pid from rfl,offset_at]
      simp only [Geometry.offset,Geometry.address,hr,hc]
    rw [← ho,hread i hi,expected_logical cfg u p s heven hcover i hi]
    change Cell.mk .int (logicalResult cfg u p s (Geometry.outRow (geometry p) pid i)
      (Geometry.outColumn (geometry p) pid i))=_
    rw [hr,hc,result_preserved cfg u p initial s ha hb hscale hout]
  · intro r o h
    apply hframe r o
    rcases h with h|h
    · exact Or.inl h
    · right
      intro i hi
      let x := Geometry.index (geometry p) pid i hi
      have hx : Geometry.Owns (geometry p) pid x := ⟨i,rfl,rfl⟩
      have ho : outputOffset p s i=Geometry.address (geometry p) x := by
        exact offset_at p current pid i
      rw [ho]
      exact h x hx

theorem grid_correct (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int) (p : Parameters)
    (hgeo : Geometry.Valid (geometry p))
    (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner) (hcover : p.inner≤p.steps*p.blockInner)
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hscale : p.scale≠p.output)
    (initial : State Word) :
    ∃ final, grid rest cfg u p initial=some final ∧
      GridExecution.Post p.output (Geometry.address (geometry p))
        (fun x : LogicalIndex p => Cell.mk .int (logicalResult cfg u p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,hread,hframe⟩ := GridExecution.compose p.output (Geometry.address (geometry p))
    hgeo.writable (fun x : LogicalIndex p => Cell.mk .int (logicalResult cfg u p initial x.1.val x.2.val))
    (Geometry.Owns (geometry p)) (one rest cfg u p) initial (Geometry.count (geometry p))
    (fun pid _ current hout => single_logical rest cfg u p hgeo heven hcover ha hb hscale initial current hout pid)
    (List.range (Geometry.count (geometry p))) (fun _ h => List.mem_range.mp h) initial (fun _ _ _ => rfl)
  refine ⟨final,he,?_,?_⟩
  · intro x _
    obtain ⟨pid,hpid,hx⟩ := Geometry.covers (geometry p) hgeo x
    exact hread x ⟨pid,List.mem_range.mpr hpid,hx⟩
  · intro r o h
    exact hframe r o (h.imp id (fun h x _ => h x trivial))

#axiomsClean executionMemory
#axiomsClean correctness
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.TotalScaled
