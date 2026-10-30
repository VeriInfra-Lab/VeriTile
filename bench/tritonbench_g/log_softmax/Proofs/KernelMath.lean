import bench.tritonbench_g.log_softmax.Proofs.Kernel
import bench.tritonbench_g.log_softmax.Proofs.Real
import bench.tritonbench_g.log_softmax.Proofs.Congruence
import bench.tritonbench_g.log_softmax.Proofs.Address

namespace VeriTile.Bench.Repairs.LogSoftmax.KernelMath
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section
open Classical
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

/-- Source-level reduction results are row scalars. A fixed complete tree is
chosen for each reduction; FP equivalence preserves that chosen tree exactly. -/
def uniform (plan : Plan N C) : Plan N C where
  maxTree := fun _ => plan.maxTree 0
  forwardSum := fun _ => plan.forwardSum 0
  backwardSum := fun _ => plan.backwardSum 0
  maxComplete := fun _ => plan.maxComplete 0
  forwardComplete := fun _ => plan.forwardComplete 0
  backwardComplete := fun _ => plan.backwardComplete 0
  covers := plan.covers
  backwardSchedule := .separate

def forwardEngine (ops : Ops A) (plan : Plan N C) : Kernel.Engine A C :=
  ⟨ops,(plan.maxTree 0).eval ops.maximum,(plan.forwardSum 0).eval ops.add⟩

def backwardEngine (ops : Ops A) (plan : Plan N C) : Kernel.Engine A C :=
  ⟨ops,(plan.maxTree 0).eval ops.maximum,(plan.backwardSum 0).eval ops.add⟩

def row [Inhabited A] (X : RegionName) (N K BM : Nat) (s : State A) (i : Fin BM) :
    Nat → A :=
  fun j => (s.mem X ((s.pids 0*BM+i.val)*N*K+j*K+s.pids 1)).read .real

theorem loaded_row [Inhabited A] (padding : A) (X : RegionName)
    (M N K BM C : Nat) (s : State A) (i : Fin BM)
    (hm : s.pids 0*BM+i.val<M) (hk : s.pids 1<K) (j : Fin C) :
    Kernel.loaded padding X M N K BM C s (i,j,PUnit.unit) =
      if j.val<N then row X N K BM s i j.val else padding := by
  by_cases hj : j.val<N
  · have sm : min (s.pids 0*BM+i.val) (M-1)=s.pids 0*BM+i.val :=
      Nat.min_eq_left (by omega)
    have sn : min j.val (N-1)=j.val := Nat.min_eq_left (by omega)
    have sk : min (s.pids 1) (K-1)=s.pids 1 := Nat.min_eq_left (by omega)
    simp [Kernel.loaded,Kernel.active,Kernel.address,hm,hj,hk,sm,sn,sk,row]
  · simp [Kernel.loaded,Kernel.active,hj]

theorem forward_graph [Inhabited A] (ops : Ops A) (plan : Plan N C) (X : RegionName)
    (M K BM : Nat) (s : State A) (i : TileIndex [BM,C])
    (active : Kernel.active M N K BM (s.pids 0) (s.pids 1) i) :
    Kernel.forwardValue (forwardEngine ops plan) X M N K BM s i =
      forward ops (uniform plan) (row X N K BM s i.1) i.2.1.val := by
  have loads : (fun j : Fin C =>
      Kernel.loaded ops.negativeInfinity X M N K BM C s (i.1,j,PUnit.unit)) =
      forwardLoad ops N (row X N K BM s i.1) := by
    funext j
    rw [loaded_row _ _ _ _ _ _ _ _ _ active.1 active.2.2 j]
    rfl
  have atI := loaded_row ops.negativeInfinity X M N K BM C s i.1 active.1 active.2.2 i.2.1
  rcases i with ⟨a,b,u⟩
  cases u
  simp only [Kernel.forwardValue,forwardEngine,loads]
  simp only [forward,cast32,rowMax,uniform,denominator,exponentials,shifted,forwardLoad]
  simp only [atI,if_pos active.2.1]
  have exps : (fun j : Fin C => ops.exp (ops.subtract
      (Kernel.loaded ops.negativeInfinity X M N K BM C s (a,j,PUnit.unit))
      ((plan.maxTree 0).eval ops.maximum (forwardLoad ops N (row X N K BM s a))))) =
      exponentials ops (uniform plan) (row X N K BM s a) := by
    funext j
    rw [congrFun loads j]
    rfl
  rw [exps]
  rfl

theorem backward_graph [Inhabited A] (ops : Ops A) (plan : Plan N C) (Y G : RegionName)
    (M K BM : Nat) (s : State A) (i : TileIndex [BM,C])
    (active : Kernel.active M N K BM (s.pids 0) (s.pids 1) i) :
    Kernel.backwardValue (backwardEngine ops plan) Y G M N K BM s i =
      backward ops (uniform plan) (row Y N K BM s i.1) (row G N K BM s i.1) i.2.1.val := by
  have loads : (fun j : Fin C =>
      Kernel.loaded ops.zero G M N K BM C s (i.1,j,PUnit.unit)) =
      gradientLoad ops N (row G N K BM s i.1) := by
    funext j
    rw [loaded_row _ _ _ _ _ _ _ _ _ active.1 active.2.2 j]
    rfl
  have gy := loaded_row ops.negativeInfinity Y M N K BM C s i.1 active.1 active.2.2 i.2.1
  have gg := loaded_row ops.zero G M N K BM C s i.1 active.1 active.2.2 i.2.1
  rcases i with ⟨a,b,u⟩
  cases u
  simp only [Kernel.backwardValue,backwardEngine,loads,gy,gg,if_pos active.2.1]
  rfl

/-- Input interpretation is pointwise binding to the actual active initial
memory. It is independent of either source kernel or its output. -/
def InputValues (s : State (WithBot ℝ)) (X : RegionName) (M N K : Nat)
    (xs : Nat → ℝ) : Prop :=
  ∀ i,i<M*N*K → (s.mem X i).read .real=some (xs i)

theorem row_binding (X : RegionName) (M N K BM : Nat)
    (s : State (WithBot ℝ)) (xs : Nat → ℝ) (binding : InputValues s X M N K xs)
    (i : Fin BM) (hm : s.pids 0*BM+i.val<M) (hk : s.pids 1<K)
    (j : Nat) (hj : j<N) :
    row X N K BM s i j=some (xs ((s.pids 0*BM+i.val)*N*K+j*K+s.pids 1)) :=
  binding _ (VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress.non_inner_bound hm hj hk)

theorem forward_math (plan : Plan N C) (X : RegionName) (M K BM : Nat)
    (s : State (WithBot ℝ)) (xs : Nat → ℝ) (binding : InputValues s X M N K xs)
    (i : TileIndex [BM,C]) (active : Kernel.active M N K BM (s.pids 0) (s.pids 1) i) :
    Kernel.forwardValue (forwardEngine RealProof.ops plan) X M N K BM s i =
      some (RealProof.targetForward
        (fun j : Fin N => xs ((s.pids 0*BM+i.1.val)*N*K+j.val*K+s.pids 1))
        ⟨i.2.1.val,active.2.1⟩) := by
  rw [forward_graph _ _ _ _ _ _ _ _ active]
  rw [forward_congr_active RealProof.ops (uniform plan) _
    (fun j => some (xs ((s.pids 0*BM+i.1.val)*N*K+j*K+s.pids 1)))
    (row_binding X M N K BM s xs binding i.1 active.1 active.2.2) _ active.2.1]
  exact RealProof.forward_target (uniform plan) _ ⟨i.2.1.val,active.2.1⟩

theorem backward_math (plan : Plan N C) (Y G : RegionName) (M K BM : Nat)
    (s : State (WithBot ℝ)) (ys gs : Nat → ℝ)
    (bindingY : InputValues s Y M N K ys) (bg : InputValues s G M N K gs)
    (i : TileIndex [BM,C]) (active : Kernel.active M N K BM (s.pids 0) (s.pids 1) i) :
    Kernel.backwardValue (backwardEngine RealProof.ops plan) Y G M N K BM s i =
      some (RealProof.targetBackward N
        (fun j => ys ((s.pids 0*BM+i.1.val)*N*K+j*K+s.pids 1))
        (fun j => gs ((s.pids 0*BM+i.1.val)*N*K+j*K+s.pids 1)) i.2.1.val) := by
  rw [backward_graph _ _ _ _ _ _ _ _ _ active]
  rw [backward_congr_active RealProof.ops (uniform plan) _ _
    (fun j => some (ys ((s.pids 0*BM+i.1.val)*N*K+j*K+s.pids 1)))
    (fun j => some (gs ((s.pids 0*BM+i.1.val)*N*K+j*K+s.pids 1)))
    (row_binding Y M N K BM s ys bindingY i.1 active.1 active.2.2)
    (row_binding G M N K BM s gs bg i.1 active.1 active.2.2) _ active.2.1]
  exact RealProof.backward_target (uniform plan) _ _ _

end
end VeriTile.Bench.Repairs.LogSoftmax.KernelMath
