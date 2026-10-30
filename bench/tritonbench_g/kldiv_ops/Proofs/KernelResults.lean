import bench.tritonbench_g.kldiv_ops.IO
import bench.tritonbench_g.kldiv_ops.Proofs.FPGraph
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardCoverage

namespace VeriTile.Bench.Repairs.KldivOps.KernelResults
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem offset_lt (p : Parameters) (row col : Nat) (hr : row<p.rows) (hc : col<p.cols) :
    row*p.cols+col<p.rows*p.cols := by
  calc
    row*p.cols+col < (row+1)*p.cols := by nlinarith
    _ ≤ p.rows*p.cols := Nat.mul_le_mul_right _ (by omega)

theorem coordinates (p : Parameters) (i : Nat) (hi : i<p.rows*p.cols) :
    0<p.cols ∧ i/p.cols<p.rows ∧ i%p.cols<p.cols ∧ (i/p.cols)*p.cols+i%p.cols=i := by
  have hc : 0<p.cols := by
    by_contra h
    have hz : p.cols=0 := by omega
    simp [hz] at hi
  exact ⟨hc,(Nat.div_lt_iff_lt_mul hc).mpr hi,Nat.mod_lt i hc,
    by simpa only [Nat.mul_comm] using Nat.div_add_mod i p.cols⟩

/-- Exact typed outputs of the existing complete forward source. The ordered
scalar/reduction expressions contain only initial reads, never a source runner. -/
theorem forward (M : Algebra α) (p : Parameters) (flag : Bool) (red : Nat) (eps : α)
    (initial : State α) (block : 0<p.block) (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    ∃ final, Program.grid M p flag red eps initial=some final ∧
      (∀ i,i<Interface.outputLength p red → final.mem p.O i=Cell.mk .real
        (if red=0 then Tile.scalar M flag eps ((initial.mem p.Y i).read .real) ((initial.mem p.T i).read .real)
         else FPGraph.row M p.cols p.block flag eps
           (fun j => (initial.mem p.Y (i*p.cols+j)).read .real)
           (fun j => (initial.mem p.T (i*p.cols+j)).read .real))) ∧
      (∀ (r : RegionName) (i : Nat), (r ≠ p.O ∨ Interface.outputLength p red ≤ i) → final.mem r i = initial.mem r i) := by
  obtain ⟨final,ran,memory⟩ := Program.grid_complete M p flag red eps initial block
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    rw [memory]
    by_cases mode : red=0
    · subst red
      have bound : i<p.rows*p.cols := hi
      obtain ⟨_,hr,hc,idx⟩ := coordinates p i bound
      have output := Program.grid_value M p flag eps (List.range p.rows) initial.mem
        (i/p.cols) (i%p.cols) (List.mem_range.mpr hr) block hc hy ht
      rw [idx] at output
      simpa only [if_pos rfl] using output
    · have bound : i<p.rows := by simpa only [Interface.outputLength,if_neg mode] using hi
      simpa only [if_neg mode] using FPGraph.grid_binding M p flag red eps (List.range p.rows)
        initial.mem i (List.mem_range.mpr bound) mode hy ht
  · intro r i outside
    rw [memory]
    apply Program.grid_frame
    intro row member written
    have hr := List.mem_range.mp member
    rcases written with ⟨region,position⟩
    rcases outside with different | tail
    · exact different region
    · by_cases mode : red=0
      · simp only [mode,if_true] at position
        have bound : i<p.rows*p.cols := lt_of_lt_of_le position.2
          (by simpa only [Nat.add_mul,Nat.one_mul] using Nat.mul_le_mul_right p.cols (show row+1 ≤ p.rows by omega))
        have htail : p.rows*p.cols ≤ i := by simpa only [Interface.outputLength,if_pos mode] using tail
        exact not_lt_of_ge htail bound
      · simp only [mode,if_false] at position
        subst i
        simp only [Interface.outputLength,if_neg mode] at tail
        omega

theorem backward (M : Algebra α) (p : Parameters) (flag : Bool)
    (initial : State α) (block : 0<p.block) (ht : p.T≠p.O) :
    ∃ final, Backward.grid M p flag initial=some final ∧
      (∀ i,i<p.rows*p.cols → final.mem p.O i=Cell.mk .real
        (Backward.scalar M flag ((initial.mem p.T i).read .real))) ∧
      (∀ (r : RegionName) (i : Nat), (r ≠ p.O ∨ p.rows*p.cols ≤ i) → final.mem r i = initial.mem r i) := by
  obtain ⟨final,ran,memory⟩ := Backward.grid_complete M p flag initial block
  refine ⟨final,ran,?_,?_⟩
  · intro i hi
    obtain ⟨_,hr,hc,idx⟩ := coordinates p i hi
    rw [memory]
    have output := Backward.grid_value M p flag (List.range p.rows) initial.mem
      (i/p.cols) (i%p.cols) (List.mem_range.mpr hr) block hc ht
    simpa only [idx] using output
  · intro r i outside
    rw [memory]
    apply Backward.grid_frame
    intro row member
    rcases outside with different | tail
    · exact .inl different
    · right;right
      exact le_trans (by simpa only [Nat.add_mul,Nat.one_mul] using
        Nat.mul_le_mul_right p.cols (show row+1 ≤ p.rows by have := List.mem_range.mp member; omega)) tail
end
end VeriTile.Bench.Repairs.KldivOps.KernelResults
