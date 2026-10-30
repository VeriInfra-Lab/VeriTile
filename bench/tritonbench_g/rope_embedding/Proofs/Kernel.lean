import bench.tritonbench_g.rope_embedding.Proofs.Bridge
import bench.tritonbench_g.rope_embedding.Proofs.NativeGraph
namespace VeriTile.Bench.Repairs.RopeEmbedding.Kernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

/-- Exactly the FP32 cells written by one declared (row, group) program. -/
def Written (r : Request) (row group : Nat) (region : RegionName) (o : Nat) : Prop :=
  region="Q" ∧ ∃ h c, group*4≤h ∧ h<min (group*4+4) r.heads ∧
    c<2*r.half ∧ o=r.offset row h c

/-- Output polynomial/word graph and every unwritten cell, including odd tails,
other head groups, other rows, coefficient tables, and allocation padding. -/
structure Post {α : Type} (r : Request) (row group : Nat) (s final : State α)
    (value : Nat→Nat→α) : Prop where
  values : ∀ h c,group*4≤h → h<min (group*4+4) r.heads → c<2*r.half →
    final.mem "Q" (r.offset row h c)=.mk .real (value h c)
  frame : ∀ region o,¬Written r row group region o → final.mem region o=s.mem region o

/-- Successful execution of the complete source: the two Q values are loaded
before either store, and all heads in the group run in the original order. -/
theorem executes {α : Type} [Inhabited α] (op : Operations α) (r : Request)
    (back : Bool) (s : State α) (row group : Nat) (hr : row<r.rows) (hg : group<r.groups) :
    ∃ final, FP.Structural.exec op.backend (Source.source (parameters r) back)
        (programState s row group)=some final ∧
      Post r row group s final (fun h c => scalarResult op r back
        (fun o => (s.mem "Q" o).read .real) (fun o => (s.mem "C" o).read .real)
        (fun o => (s.mem "S" o).read .real) row h c) := by
  have gh := group_head r group hg
  obtain ⟨final,he,hv,hf⟩ := Source.complete_correctness op.backend (parameters r) back
    (programState s row group) (by simpa [programState,parameters] using Nat.le_of_lt gh)
  refine ⟨final,he,⟨?_,?_⟩⟩
  · intro h c hlo hhi hc
    let j := if c<r.half then c else c-r.half
    have hj : j<r.half := by dsimp [j]; split <;> omega
    let i : Lane r := (⟨j,lt_of_lt_of_le hj (block_covers r)⟩,PUnit.unit)
    let side := decide (r.half≤c)
    let l : Location r := (row,group,h,side,i)
    have col : column l=c := by
      by_cases h : c<r.half
      · simp [column,l,lane,RopeEmbedding.side,side,i,j,h,show ¬r.half≤c from by omega]
      · simp [column,l,lane,RopeEmbedding.side,side,i,j,h,show r.half≤c from by omega]
    have eq := hv h (by simpa [programState] using hlo)
      (by simpa [programState,parameters] using hhi) side i hj
    have sourceEq : final.mem "Q" (r.offset row h (column l))=
        .mk .real (sourceValue op r back s l) := by
      simpa only [sourceValue,programState,source_offset,column,
        RopeEmbedding.row,RopeEmbedding.group,head,RopeEmbedding.side,side,l,lane,↓reduceIte] using eq
    have act : active l := ⟨hr,lt_of_lt_of_le hhi (min_le_right _ _),hj⟩
    rw [source_value op r back s l act] at sourceEq
    simpa only [col,RopeEmbedding.row,head,l] using sourceEq
  · intro region o outside
    apply hf region o
    by_cases hregion : region="Q"
    · right
      intro h hlo hhi side i hi equal
      apply outside
      refine ⟨hregion,h,i.1.val+if side then r.half else 0,?_,?_,?_,?_⟩
      · simpa [programState] using hlo
      · simpa [programState,parameters] using hhi
      · change i.1.val<r.half at hi
        cases side <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
      · simpa [programState,source_offset] using equal
    · exact Or.inl hregion

/-- The written set excludes an odd final channel. -/
theorem tail_not_written (r : Request) (row group h c : Nat)
    (hh : h<r.heads) (hc : c<r.dimension) (tail : 2*r.half≤c) :
    ¬Written r row group "Q" (r.offset row h c) := by
  rintro ⟨_,k,j,_,hk,hj,eq⟩
  have dim : 2*r.half≤r.dimension := by unfold Request.half; omega
  have inj := offset_injective r hh hc (lt_of_lt_of_le hk (min_le_right _ _))
    (by omega) eq
  omega

/-- Every rotated coordinate has a valid group and lane, including partial
last groups and padded blocks. This is kernel coverage, not host submission. -/
theorem complete_coverage (r : Request) (row h c : Nat)
    (hr : row<r.rows) (hh : h<r.heads) (hc : c<2*r.half) :
    ∃ g,g<r.groups ∧ Written r row g "Q" (r.offset row h c) := by
  refine ⟨h/4,?_,rfl,h,c,?_,?_,hc,rfl⟩
  · unfold Request.groups; omega
  · omega
  · exact lt_min (by omega) hh

/-- Distinct program coordinates cannot overwrite each other's Q input.
Together with the coefficient frame, this justifies reuse for any grid order. -/
theorem other_program_not_written (r : Request) (row group peer head col : Nat)
    (hh : head<r.heads) (hc : col<r.dimension)
    (other : peer≠row ∨ head/4≠group) :
    ¬Written r row group "Q" (r.offset peer head col) := by
  rintro ⟨_,h,c,hlo,hhi,active,eq⟩
  have dim : 2*r.half≤r.dimension := by unfold Request.half; omega
  have inj := offset_injective r hh hc (lt_of_lt_of_le hhi (min_le_right _ _))
    (by omega) eq
  have upper := lt_of_lt_of_le hhi (min_le_left _ _)
  rcases other with other | other <;> omega

#axiomsClean executes
#axiomsClean tail_not_written
#axiomsClean complete_coverage
#axiomsClean other_program_not_written
end
end VeriTile.Bench.Repairs.RopeEmbedding.Kernel
