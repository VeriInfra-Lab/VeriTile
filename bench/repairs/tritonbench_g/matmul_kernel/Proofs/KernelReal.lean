import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelScalar
import bench.repairs.tritonbench_g.matmul_kernel.Spec
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelReal
open VeriTile.Triton FP.Structural KernelSource KernelScalar RealSum
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

def scalar : Algebra ℝ := {fallback 0 with literal:=fun _ _ x=>x}
def dot : LaneDot ℝ := fun _ _ k a b z=>z+sum (fun e=>if he:e<k then a ⟨e,he⟩*b ⟨e,he⟩ else 0) k

/-- The independent Real target reads only initial logical matrix values. -/
def target (K : Nat) (a b : Nat → Nat → ℝ) (r n : Nat) : ℝ :=
  ∑ k∈Finset.range K,a r k*b k n

theorem fold_sum (f : Nat → ℝ) (B cur stop : Nat) (h : cur≤stop) (z : ℝ) :
    foldRange (fun q x=>x+sum (fun e=>f (q*B+e)) B) cur stop 1 (z+sum f (cur*B))=
      z+sum f (stop*B) := by
  by_cases hc : cur<stop
  · rw [foldRange]
    simp only [show ¬(1:Nat)=0 by decide,hc,if_false,if_true]
    have eq : z+sum f (cur*B)+sum (fun e=>f (cur*B+e)) B=z+sum f ((cur+1)*B) := by
      rw [Nat.add_mul,Nat.one_mul,sum_add]
      ring
    rw [eq]
    exact fold_sum f B (cur+1) stop (by omega) z
  · have same : cur=stop := by omega
    subst cur
    simp [foldRange]
termination_by stop-cur
decreasing_by omega

theorem reference_target (p : Config) (bk : 0<p.BK) (a b : Nat → Nat → ℝ) (r n : Nat) :
    reference scalar dot p a b r n=target p.K a b r n := by
  let f : Nat → ℝ := fun k=>if k<p.K then a r k*b k n else 0
  have update (q : Nat) (z : ℝ) :
      dot (r%p.BM) (n%p.BN) p.BK
        (fun e=>(if q*p.BK+e.val<p.K then a r (q*p.BK+e.val) else 0))
        (fun e=>(if q*p.BK+e.val<p.K then b (q*p.BK+e.val) n else 0)) z=
        z+sum (fun e=>f (q*p.BK+e)) p.BK := by
    unfold dot
    congr 1
    apply sum_congr
    intro e he
    simp only [dif_pos he,f]
    split_ifs <;> simp
  unfold reference
  simp only [scalar,fallback,id_eq,update]
  have fold:=fold_sum f p.BK 0 (ceil p.K p.BK) (Nat.zero_le _) 0
  simp only [Nat.zero_mul,sum,add_zero,zero_add] at fold
  rw [fold]
  change sum (fun k=>if k<p.K then a r k*b k n else 0) (ceil p.K p.BK*p.BK)=_
  rw [sum_padding,Nat.min_eq_left (MatmulKernel.ceil_cover _ _ bk),sum_finset]
  rfl

theorem launch_correct (p : Config) (hc : p.Valid) (s : State ℝ) :
    ∃ t,KernelSource.launch scalar (dotAcc dot) p s=some t ∧
      (∀ w,p.Active w → t.mem "C" (p.C.addr w.1 w.2)=stored
        (target p.K (fun r k=>read (s.mem "A" (p.A.addr r k)))
          (fun k n=>read (s.mem "B" (p.B.addr k n))) w.1 w.2)) ∧
      (∀ (region : RegionName) offset,(region≠"C" ∨ ∀ w,p.Active w → offset≠p.C.addr w.1 w.2) →
        t.mem region offset=s.mem region offset) := by
  obtain ⟨t,he,hr,hf⟩:=KernelSource.launch_correct scalar (dotAcc dot) p hc s
  refine ⟨t,he,?_,hf⟩
  intro w hw
  rw [hr w hw,point_reference _ _ _ hc s w hw,reference_target _ hc.bk]

#specNonCircular target avoiding [KernelSource.source,KernelSource.outputValue,KernelSource.point,KernelSource.launch,reference]
#axiomsClean reference_target
#axiomsClean launch_correct
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelReal
