import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReference
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Real
namespace VeriTile.Bench.Repairs.Conv2d.KernelReal
open VeriTile.Triton FP.Structural KernelNumeric
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 16384

def operations : Operations ℝ := ⟨0,id,fun a b carry=>carry+∑ k : Fin 32,a k*b k⟩

@[simp] theorem convert_real (f : Bool) (x : ℝ) : convert operations f x=x := by
  cases f <;> rfl

theorem fold_sum (f : Nat → ℝ) (cur stop : Nat) (h : cur≤stop) (z : ℝ) :
    foldRange (fun q acc=>acc+f q) cur stop 1 (z+∑ q∈Finset.range cur,f q)=
      z+∑ q∈Finset.range stop,f q := by
  by_cases hc : cur<stop
  · rw [foldRange]
    simp only [dif_neg (by decide : ¬(1:Nat)=0),dif_pos hc]
    have eq : (z+∑ q∈Finset.range cur,f q)+f cur=z+∑ q∈Finset.range (cur+1),f q := by
      rw [Finset.sum_range_succ];ring
    rw [eq]
    exact fold_sum f (cur+1) stop (by omega) z
  · have eq : cur=stop := by omega
    subst cur
    rw [foldRange]
    simp only [dif_neg (by decide : ¬(1:Nat)=0),dif_neg (Nat.lt_irrefl stop)]
termination_by stop-cur
decreasing_by omega


theorem fold_sum_zero (f : Nat → ℝ) (stop : Nat) (z : ℝ) :
    foldRange (fun q acc=>acc+f q) 0 stop 1 z=z+∑ q∈Finset.range stop,f q := by
  simpa using fold_sum f 0 stop (Nat.zero_le _) z

theorem block_test (cur stop : Nat) : cur<blocks stop 32 ↔ cur*32<stop := by
  unfold blocks
  omega

theorem block_fold (f : Nat → ℝ) (cur stop : Nat) (h : cur≤blocks stop 32) (z : ℝ) :
    foldRange (fun c acc=>acc+∑ k : Fin 32,f (c+k.val)) (cur*32) stop 32
      (z+∑ c∈Finset.range (cur*32),f c)=z+∑ c∈Finset.range (blocks stop 32*32),f c := by
  by_cases hc : cur<blocks stop 32
  · rw [foldRange]
    simp only [dif_neg (by decide : ¬(32:Nat)=0),dif_pos ((block_test cur stop).mp hc)]
    have eq : (z+∑ c∈Finset.range (cur*32),f c)+(∑ k : Fin 32,f (cur*32+k.val))=
        z+∑ c∈Finset.range ((cur+1)*32),f c := by
      rw [Fin.sum_univ_eq_sum_range (fun k=>f (cur*32+k)) 32,Nat.add_mul,Nat.one_mul,Finset.sum_range_add]
      ring
    rw [eq,show cur*32+32=(cur+1)*32 by omega]
    exact block_fold f (cur+1) stop (by omega) z
  · have eq : cur=blocks stop 32 := by omega
    have hn : ¬cur*32<stop := mt (block_test cur stop).mpr hc
    rw [foldRange]
    simp only [dif_neg (by decide : ¬(32:Nat)=0),dif_neg hn]
    rw [eq]
termination_by blocks stop 32-cur
decreasing_by omega


theorem channels_real (s : Shape) (m : Memory ℝ) (x : Conv2d.Cell) (h w : Nat) (z : ℝ) :
    foldRange (fun c acc=>acc+∑ k : Fin 32,Conv2d.term s m x.group x.spatial x.feature h w (c+k.val))
      0 s.cg 32 z=z+∑ c∈Finset.range s.cg,Conv2d.term s m x.group x.spatial x.feature h w c := by
  have he:=block_fold (fun c=>Conv2d.term s m x.group x.spatial x.feature h w c) 0 s.cg (Nat.zero_le _) z
  simp only [Nat.zero_mul,Finset.range_zero,Finset.sum_empty,add_zero] at he
  rw [he]
  congr 1
  symm
  apply Finset.sum_subset (Finset.range_mono (block_cover s.cg 32 (by decide)))
  intro c hc hn
  have ge : ¬c<s.cg := by simpa using hn
  simp [Conv2d.term,refA,refB,ge,realISA]


/-- Exact independent grouped cross-correlation under ideal Real arithmetic.
No finite-value, native-to-Real equality, or numerical-conformance premise is
needed, and all four public precision flags retain their original meaning. -/
theorem reference_real (s : Shape) (m : Memory ℝ) (f : Flags) (x : Conv2d.Cell)
    (hx : s.Active x) : reference operations s f m x=mathematical s m x := by
  have upd (h w c : Nat) (acc : ℝ) :
      operations.lower32
        (fun e=>convert operations f.fp16 (refA s (arrayISA operations.zero) m x.group x.spatial h w (c+e.val)))
        (fun e=>convert operations f.fp16 (refB s (arrayISA operations.zero) m x.group x.feature h w (c+e.val))) acc=
      acc+∑ k : Fin 32,Conv2d.term s m x.group x.spatial x.feature h w (c+k.val) := by
    simp only [convert_real]
    rfl
  unfold reference
  simp_rw [upd]
  simp only [operations]
  simp_rw [channels_real]
  simp_rw [fold_sum_zero]
  simp only [zero_add]
  have cut : (∑ h∈Finset.range s.kh,∑ w∈Finset.range s.kw,∑ c∈Finset.range s.cg,
      Conv2d.term s m x.group x.spatial x.feature h w c)=mathematical s m x := by
    simp_rw [Finset.sum_comm (s:=Finset.range s.kw) (t:=Finset.range s.cg)]
    rw [Finset.sum_comm]
    unfold mathematical
    apply Finset.sum_congr rfl
    intro c hc
    apply Finset.sum_congr rfl
    intro h hh
    apply Finset.sum_congr rfl
    intro w hw
    have clt:=Finset.mem_range.mp hc
    simp [Conv2d.term,refA,refB,clt,hx.2.2,realISA]
  exact cut

#axiomsClean block_fold
#axiomsClean reference_real
end
end VeriTile.Bench.Repairs.Conv2d.KernelReal
