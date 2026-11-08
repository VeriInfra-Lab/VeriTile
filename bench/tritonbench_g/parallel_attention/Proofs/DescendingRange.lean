import bench.tritonbench_g.parallel_attention.Proofs.Support.FPBackwardGraph
import VeriTile.Triton.Float.StructuralSafetyLoops
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.DescendingRange
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Count precisely the reached signed source indices; the executable loop
retains its signed counter and original positive magnitude of the stride. -/
theorem clock (M : Algebra α) (index : RegName) (body : List ComputeStmt)
    (offset stop : Int) (stride count : Nat) (positive : 0<stride)
    (active : ∀ n : Nat, offset+(n : Int)*(stride : Int)<stop ↔ n<count)
    (I : Nat → State α → Prop)
    (advance : ∀ n, n<count → ∀ s, I n s →
      ∃ t, run M body (s.setReg index .int [] (fun _=>offset+(n : Int)*(stride : Int)))=some t ∧ I (n+1) t)
    (start : Nat) (bound : start≤count) (s : State α) (initial : I start s) :
    ∃ t, rangeInt M index (offset+(start : Int)*(stride : Int)) stop stride body s=some t ∧ I count t := by
  by_cases live : start<count
  · obtain ⟨middle,ran,maintained⟩:=advance start live s initial
    obtain ⟨final,finished,invariant⟩:=clock M index body offset stop stride count positive active I advance
      (start+1) (by omega) middle maintained
    refine ⟨final,?_,invariant⟩
    rw [rangeInt]
    simp only [Nat.ne_of_gt positive,↓reduceIte,(active start).mpr live,ran,Option.bind_some]
    have next : offset+(start : Int)*(stride : Int)+(stride : Int)=
        offset+((start+1 : Nat) : Int)*(stride : Int) := by simp [Nat.cast_add,add_mul,add_assoc]
    rw [next]
    exact finished
  · have last : start=count:=by omega
    subst start
    refine ⟨s,?_,initial⟩
    rw [rangeInt]
    simp [Nat.ne_of_gt positive,active]
termination_by count-start
decreasing_by omega

theorem clock_safe (M : Algebra α) (bounds : RegionName → Nat) (index : RegName)
    (body : List ComputeStmt) (offset stop : Int) (stride count : Nat)
    (active : ∀ n : Nat, offset+(n : Int)*(stride : Int)<stop ↔ n<count)
    (I : Nat → State α → Prop)
    (bodySafe : ∀ n, n<count → ∀ s, I n s →
      Safety.traceSafeList M bounds body (s.setReg index .int [] (fun _=>offset+(n : Int)*(stride : Int))))
    (preserved : ∀ n, n<count → ∀ s, I n s → ∀ t,
      run M body (s.setReg index .int [] (fun _=>offset+(n : Int)*(stride : Int)))=some t → I (n+1) t)
    (start : Nat) (bound : start≤count) (s : State α) (initial : I start s) :
    Safety.rangeIntSafe M bounds index (offset+(start : Int)*(stride : Int)) stop stride body s := by
  let P:=fun q t=>∃ n : Nat,n≤count ∧ q=offset+(n : Int)*(stride : Int) ∧ I n t
  apply Safety.rangeIntSafe_of_invariant M bounds index stop stride body P
  · intro q live t reached
    obtain ⟨n,hn,rfl,state⟩:=reached
    exact bodySafe n ((active n).mp live) t state
  · intro q live t reached u ran
    obtain ⟨n,hn,rfl,state⟩:=reached
    have next : offset+(n : Int)*(stride : Int)+(stride : Int)=
        offset+((n+1 : Nat) : Int)*(stride : Int) := by simp [Nat.cast_add,add_mul,add_assoc]
    exact ⟨n+1,by have := (active n).mp live;omega,next,preserved n ((active n).mp live) t state u ran⟩
  · exact ⟨start,bound,rfl,initial⟩

def lower (c : BackwardConfig) : Int :=
  -(((c.T+c.BTS-1)/c.BTS*c.BTS : Nat) : Int)+(c.BTS : Int)
def upper (c : BackwardConfig) : Int :=
  (c.BTS : Int)-(((c.i_c+1)*c.BTL : Nat) : Int)

theorem ceil_active (n size stride : Nat) (positive : 0<stride) :
    n<(size+stride-1)/stride ↔ n*stride<size := by
  have division:=Nat.le_div_iff_mul_le (x:=n+1) (y:=size+stride-1) positive
  have successor : (n+1)*stride=n*stride+stride:=by simp [Nat.add_mul]
  rw [successor] at division
  omega

theorem mirrored_active (top next stride n : Nat) (positive : 0<stride) :
    -(top : Int)+(stride : Int)+(n : Int)*(stride : Int)<(stride : Int)-(next : Int) ↔
      n<(top-next+stride-1)/stride := by
  rw [ceil_active _ _ _ positive]
  have product : (n : Int)*(stride : Int)=((n*stride : Nat) : Int):=by norm_cast
  rw [product]
  omega

theorem mirrored_index (top stride n : Nat) (last : n*stride+stride≤top) :
    -(-(top : Int)+(stride : Int)+(n : Int)*(stride : Int))=(top-stride-n*stride : Nat) := by
  have product : (n : Int)*(stride : Int)=((n*stride : Nat) : Int):=by norm_cast
  rw [product,Nat.cast_sub (by omega : n*stride≤top-stride),Nat.cast_sub (by omega : stride≤top)]
  omega

theorem source_active (c : BackwardConfig) (positive : 0<c.BTS) (n : Nat) :
    lower c+(n : Int)*(c.BTS : Int)<upper c ↔ n<descendingCount c :=
  mirrored_active _ _ _ _ positive

theorem source_index (c : BackwardConfig) (positive : 0<c.BTS) (n : Nat)
    (live : n<descendingCount c) :
    -(lower c+(n : Int)*(c.BTS : Int))=(descendingStart c n : Int) := by
  have reached : n*c.BTS<(c.T+c.BTS-1)/c.BTS*c.BTS := by
    have bound:=(ceil_active n _ c.BTS positive).mp live
    exact lt_of_lt_of_le bound (Nat.sub_le _ _)
  have block : n<(c.T+c.BTS-1)/c.BTS:= (Nat.mul_lt_mul_right positive).mp reached
  have last := Nat.mul_le_mul_right c.BTS (show n+1≤(c.T+c.BTS-1)/c.BTS by omega)
  rw [Nat.add_mul,Nat.one_mul] at last
  exact mirrored_index _ _ _ last

#axiomsClean clock
#axiomsClean clock_safe
#axiomsClean source_active
#axiomsClean source_index
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.DescendingRange
