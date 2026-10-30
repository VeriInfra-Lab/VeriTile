import bench.tritonbench_g.softmax_flaggems.Proofs.RealOneTile
import bench.tritonbench_g.softmax_flaggems.Proofs.TraceReadback
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridMetadata

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

/-- Mathematical scope is a property of values, separate from wrapper admission. -/
def Defined (q : Nat → WithBot ℝ) (d : Dimensions) : Prop :=
  ∀ row, row<d.M → ∀ col, col<d.K → Target.Defined (Target.column q d.N d.K row col)

def Input (X : RegionName) (q : Nat → WithBot ℝ) (s : State (WithBot ℝ)) : Prop :=
  ∀ o, FPReference.read s X o=q o

def cells (q : Nat → WithBot ℝ) (N K : Nat) (_ : RegionName) (o : Nat) : Cell (WithBot ℝ) :=
  .mk .real ((Target.flatForward q N K o : ℝ) : WithBot ℝ)

theorem noninner_ratio_flat (s : State (WithBot ℝ)) (X : RegionName) (q : Nat → WithBot ℝ)
    (input : Input X q s) (row N K k n : Nat) (hn : n<N) (hk : k<K) :
    Target.mass (noninnerInput s X row N K k n) /
      Target.denominator (fun j : Fin N => noninnerInput s X row N K k j.val) =
      Target.flatForward q N K (Target.address N K row k n) := by
  obtain ⟨hr,hc⟩ := Target.coordinates N K row k n (by omega) (by omega) hn hk
  dsimp only [Input] at input
  unfold Target.flatForward
  rw [hr,hc]
  simp only [Target.column,Target.address,noninnerInput,if_pos hk,input]
  rfl

theorem inner_ratio_flat (s : State (WithBot ℝ)) (X : RegionName) (q : Nat → WithBot ℝ)
    (input : Input X q s) (row N n : Nat) (hn : n<N) :
    Target.mass (innerInput s X row N n) /
      Target.denominator (fun j : Fin N => innerInput s X row N j.val) =
      Target.flatForward q N 1 (row*N+n) := by
  simpa [noninnerInput,innerInput,Target.address] using
    noninner_ratio_flat s X q input row N 1 0 n hn (by decide)

theorem input_preserved (X Y : RegionName) (q : Nat → WithBot ℝ)
    (t : FPTrace.Tile (WithBot ℝ)) (region : t.region=Y) (separate : X≠Y)
    (s : State (WithBot ℝ)) (input : Input X q s) : Input X q (t.execute s) := by
  intro o
  change ((t.execute s).mem X o).read .real=q o
  rw [FPTrace.Tile.frame]
  · exact input o
  · intro hit
    obtain ⟨i,hi,he⟩ := (FPTrace.Tile.mem_events t (X,o)).mp hit
    exact separate ((congrArg Prod.fst he).trans region)

theorem noninner_defined (q : Nat → WithBot ℝ) (d : Dimensions) (defined : Defined q d)
    (s : State (WithBot ℝ)) (X : RegionName) (input : Input X q s)
    (row k : Nat) (hr : row<d.M) (hk : k<d.K) :
    Target.Defined (fun n : Fin d.N => noninnerInput s X row d.N d.K k n.val) := by
  have h := defined row hr k hk
  dsimp only [Input] at input
  simpa only [Target.column,Target.address,noninnerInput,if_pos hk,input] using h

theorem inner_defined (q : Nat → WithBot ℝ) (d : Dimensions) (defined : Defined q d)
    (s : State (WithBot ℝ)) (X : RegionName) (input : Input X q s)
    (row : Nat) (hr : row<d.M) (hK : d.K=1) :
    Target.Defined (fun n : Fin d.N => innerInput s X row d.N n.val) := by
  have h := noninner_defined q d defined s X input row 0 hr (by omega)
  simpa only [hK,noninnerInput,innerInput,Nat.zero_lt_one,if_pos,Nat.mul_one,Nat.add_zero] using h

#axiomsClean noninner_ratio_flat
#axiomsClean inner_ratio_flat
#axiomsClean input_preserved
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.RealForward
