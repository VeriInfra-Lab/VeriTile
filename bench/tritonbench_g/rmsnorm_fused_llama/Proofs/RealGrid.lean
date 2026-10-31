import bench.tritonbench_g.rmsnorm_fused_llama.Kernels
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.Mathematical_FullExecution

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.RealGrid
open VeriTile Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000

/-- Independent conventional RMS normalization, on the complete logical row. -/
def value (x w : Nat → ℝ) (row N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  x (row*N+k) * (Real.sqrt ((∑ j : Fin N, x (row*N+j.val)^2)/(N : ℝ)+eps))⁻¹ * w k

theorem row_complete (s : BlockState) (X Y W : RegionName) (N B : Nat) (eps : ℝ)
    (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) :
    ∃ t, Triton.exec (kernel X Y W N B eps).eraseDType.toAlgKernel s=some t ∧
      (∀ k : Fin N, t.readMem Y (s.pids 0*N+k.val)=
        value (s.readMem X) (s.readMem W) (s.pids 0) N eps k.val) ∧
      VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General.OutputFrame s t Y N N := by
  obtain ⟨t,he,hv,hf⟩ := VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General.real_execution
    s X Y W N N B eps hB hXY hWY
  refine ⟨t,?_,hv,hf⟩
  exact he

def setup (rows row : Nat) (s : BlockState) : BlockState :=
  {s with regs := fun _ _ _ => none
          pids := fun a => if a=0 then row else 0
          numPids := fun a => if a=0 then rows else 1}

def run (X Y W : RegionName) (rows N B : Nat) (eps : ℝ) (initial : BlockState) : Nat → Option BlockState
  | 0 => some initial
  | c+1 => (run X Y W rows N B eps initial c).bind
      (fun s => Triton.exec (kernel X Y W N B eps).eraseDType.toAlgKernel (setup rows c s))

def Frame (Y : RegionName) (count : Nat) (s t : BlockState) : Prop :=
  ∀ R o, R≠Y ∨ count≤o → t.mem R o=s.mem R o

theorem complete (X Y W : RegionName) (rows N B : Nat) (eps : ℝ) (initial : BlockState)
    (hB : 0<B) (hXY : X≠Y) (hWY : W≠Y) (count : Nat) :
    ∃ t, run X Y W rows N B eps initial count=some t ∧
      (∀ row, row<count → ∀ k, k<N → t.readMem Y (row*N+k)=
        value (initial.readMem X) (initial.readMem W) row N eps k) ∧
      Frame Y (count*N) initial t := by
  induction count with
  | zero => exact ⟨initial,rfl,by intro row hr; omega,by intro R o h; rfl⟩
  | succ c ih =>
    obtain ⟨s,hs,hval,hframe⟩ := ih
    obtain ⟨t,ht,hread,hf⟩ := row_complete (setup rows c s) X Y W N B eps hB hXY hWY
    have hx : s.mem X=initial.mem X := funext (fun o => hframe X o (Or.inl hXY))
    have hw : s.mem W=initial.mem W := funext (fun o => hframe W o (Or.inl hWY))
    refine ⟨t,by simpa [run,hs] using ht,?_,?_⟩
    · intro row hr k hk
      by_cases he : row=c
      · subst row
        simpa [setup,value,BlockState.readMem,hx,hw] using hread ⟨k,hk⟩
      · have hrow : row<c := by omega
        have hb : row*N+k<c*N := by
          calc row*N+k < row*N+N := by omega
               _=(row+1)*N := by simp [Nat.add_mul]
               _≤c*N := Nat.mul_le_mul_right N (by omega)
        have hm := hf Y (row*N+k) (Or.inr (fun j => by change row*N+k≠c*N+j.val; omega))
        have hh : t.readMem Y (row*N+k)=s.readMem Y (row*N+k) := by
          simp only [BlockState.readMem,hm]; rfl
        exact hh.trans (hval row hrow k hk)
    · intro R o ho
      have hm := hf R o (by
        rcases ho with h|h
        · exact Or.inl h
        · right
          intro k
          change o≠c*N+k.val
          have he : (c+1)*N=c*N+N := by simp [Nat.add_mul]
          have hk := k.isLt
          omega)
      rw [hm]
      apply hframe R o
      rcases ho with h|h
      · exact Or.inl h
      · exact Or.inr (le_trans (Nat.mul_le_mul_right N (by omega : c≤c+1)) h)

#axiomsClean row_complete
#axiomsClean complete
#specNonCircular value avoiding [run,kernel,Triton.exec]
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.RealGrid
