import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedResult
import bench.tritonbench_g.layer_norm_triton.Proofs.SourceForwardReal

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedReal
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph
open SourceForwardGraph (lit bin cast sum)
noncomputable section

def row (s : State ℝ) (region : RegionName) (stride r : Nat) : Nat → ℝ :=
  fun j => (s.mem region (r*stride+j)).read .real
def weight (s : State ℝ) : Nat → ℝ := fun j => (s.mem "W" j).read .real
def savedMean (s : State ℝ) (r : Nat) := (s.mem "Mean" r).read .real
def savedInverse (s : State ℝ) (r : Nat) := (s.mem "Rstd" r).read .real
def normalized (s : State ℝ) (stride r j : Nat) :=
  (row s "X" stride r j-savedMean s r)*savedInverse s r
def weightedDy (s : State ℝ) (stride r j : Nat) := row s "DY" stride r j*weight s j

theorem xhat_real (m : Model ℝ) (laws : SourceForwardReal.Laws m) (s : State ℝ)
    (stride N r j : Nat) : xhat m s stride N r j=if j<N then normalized s stride r j else 0 := by
  by_cases hj : j<N <;>
    simp [xhat,loaded,SourceGroupedGraph.inverse,normalized,row,savedMean,savedInverse,
      laws.literal,laws.mul,laws.sub,laws.cast,laws.load,hj]

theorem wdy_real (m : Model ℝ) (laws : SourceForwardReal.Laws m) (s : State ℝ)
    (stride N r j : Nat) : wdy m s stride N r j=if j<N then weightedDy s stride r j else 0 := by
  by_cases hj : j<N <;>
    simp [wdy,loaded,weightedDy,row,weight,laws.literal,laws.mul,laws.cast,hj,mul_comm]

theorem first_real (m : Model ℝ) (laws : SourceForwardReal.Laws m) (s : State ℝ)
    (stride N B r : Nat) (hb : N≤B) :
    first m s stride N B r=
      (∑ j ∈ Finset.range N,weightedDy s stride r j*normalized s stride r j)/(N : ℝ) := by
  have hv (j : Fin B) : bin m .mul (xhat m s stride N r j.val) (wdy m s stride N r j.val)=
      if j.val<N then weightedDy s stride r j.val*normalized s stride r j.val else 0 := by
    rw [laws.mul,xhat_real m laws,wdy_real m laws]
    split <;> simp [mul_comm]
  unfold first
  simp only [laws.div,laws.sum]
  simp only [hv]
  rw [laws.literal,SourceForwardReal.padded_sum N B hb
    (fun j => weightedDy s stride r j*normalized s stride r j)]

theorem second_real (m : Model ℝ) (laws : SourceForwardReal.Laws m) (s : State ℝ)
    (stride N B r : Nat) (hb : N≤B) :
    second m s stride N B r=(∑ j ∈ Finset.range N,weightedDy s stride r j)/(N : ℝ) := by
  simp only [second,laws.div,laws.sum,laws.literal,wdy_real m laws]
  rw [SourceForwardReal.padded_sum N B hb (weightedDy s stride r)]

theorem dx_real (m : Model ℝ) (laws : SourceForwardReal.Laws m) (s : State ℝ)
    (stride N B r j : Nat) (hb : N≤B) (hj : j<N) :
    dx m s stride N B r j=
      RealProof.Spec.gradient N (row s "X" stride r) (row s "DY" stride r) (weight s)
        (savedMean s r) (savedInverse s r) j := by
  simp only [dx,laws.cast,laws.mul,laws.sub,laws.add,xhat_real m laws,wdy_real m laws,if_pos hj,
    first_real m laws s stride N B r hb,second_real m laws s stride N B r hb]
  simp only [SourceGroupedGraph.inverse,laws.load]
  unfold RealProof.Spec.gradient VeriTile.Bench.Repairs.FastLayernorm.RealProof.gradientSpec
  change (weightedDy s stride r j-(normalized s stride r j*((_)/(N : ℝ))+(_)/(N : ℝ)))*savedInverse s r=_
  simp only [weightedDy,normalized]
  ring

theorem nextCarries_real (m : Model ℝ) (laws : SourceForwardReal.Laws m) (s : State ℝ)
    (stride N B group r : Nat) (a : Carries ℝ B) (i : TileIndex [B]) (hi : i.1.val<N) :
    (nextCarries m s stride N B group r a).1 i=
      row s "DY" stride r i.1.val*normalized s stride r i.1.val+(if r=group then 0 else a.1 i) ∧
    (nextCarries m s stride N B group r a).2 i=
      row s "DY" stride r i.1.val+(if r=group then 0 else a.2 i) := by
  by_cases hr : r=group <;>
    simp [nextCarries,loaded,xhat_real m laws,hi,hr,laws.cast,laws.mul,laws.add,row]

#axiomsClean dx_real
#axiomsClean nextCarries_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceGroupedReal
