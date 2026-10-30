import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeGrid
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

/-- Every lane of every original submitted plan, including guarded dummies. -/
abbrev Job (plans : List LaunchPlan.Item) := (p : Fin plans.length) × Fin (plans[p]).size
def plan {plans : List LaunchPlan.Item} (job : Job plans) : LaunchPlan.Item := plans[job.1]
def number {plans : List LaunchPlan.Item} (job : Job plans) : Nat := (plan job).base+job.2.val
def allJobs (plans : List LaunchPlan.Item) : List (Job plans) := Finset.univ.toList
def atJob {plans : List LaunchPlan.Item} (job : Job plans) (s : State NativeGraph.Expr) :=
  HostFP.gridState s (plan job).size job.2.val

def invocation (source : Nat → ComputeKernel) {plans : List LaunchPlan.Item} (job : Job plans) :
    Invocation NativeGraph.Expr where
  kernel := source (plan job).base
  backend := FP.ProfiledRewrite.engine NativeValue.algebra
  pids := fun ax => if ax=0 then job.2.val else 0
  numPids := some (fun ax => if ax=0 then (plan job).size else 1)

@[simp] theorem invocation_initial (source : Nat → ComputeKernel)
    {plans : List LaunchPlan.Item} (job : Job plans) (s : State NativeGraph.Expr) :
    (invocation source job).initial s=atJob job s := rfl

theorem plan_mem {plans : List LaunchPlan.Item} (job : Job plans) : plan job ∈ plans :=
  List.getElem_mem _

theorem number_injective (plans : List LaunchPlan.Item)
    (ordered : plans.Pairwise (fun p q => p.base+p.size≤q.base)) :
    Function.Injective (number (plans:=plans)) := by
  rintro ⟨i,li⟩ ⟨j,lj⟩ equal
  have left := li.isLt
  have right := lj.isLt
  have same : i=j := by
    apply Fin.ext
    rcases lt_trichotomy i.val j.val with before|same|after
    · have disjoint := ordered.rel_get_of_lt before
      change plans[i].base+plans[i].size≤plans[j].base at disjoint
      change plans[i].base+li.val=plans[j].base+lj.val at equal
      omega
    · exact same
    · have disjoint := ordered.rel_get_of_lt after
      change plans[j].base+plans[j].size≤plans[i].base at disjoint
      change plans[i].base+li.val=plans[j].base+lj.val at equal
      omega
  subst j
  have lanes : li=lj := Fin.ext (Nat.add_left_cancel equal)
  subst lj
  rfl

/-- Original bases are a full grid-limit apart, including before a short last
batch. This supplies ordering for every native BS, even when BS is below 16. -/
theorem build_base_spacing (total fuel base : Nat) :
    (LaunchPlan.build total fuel base).Pairwise (fun p q => p.base+Grid.limit≤q.base) := by
  induction fuel generalizing base with
  | zero => simp [LaunchPlan.build]
  | succ fuel ih =>
    simp only [LaunchPlan.build]
    split_ifs with active
    · rw [List.pairwise_cons]
      refine ⟨?_,ih _⟩
      intro q member
      exact (LaunchPlan.member_valid total fuel (base+Grid.limit) q member).2
    · exact List.Pairwise.nil

theorem tune_ordered (total : Nat) (plans : List LaunchPlan.Item)
    (spacing : plans.Pairwise (fun p q => p.base+Grid.limit≤q.base)) :
    (plans.map (AutotunePlan.tune total)).Pairwise (fun p q => p.base+p.size≤q.base) := by
  induction plans with
  | nil => simp
  | cons p rest ih =>
    obtain ⟨head,tail⟩ := List.pairwise_cons.mp spacing
    rw [List.map_cons,List.pairwise_cons]
    refine ⟨?_,ih tail⟩
    intro selected selectedMem
    obtain ⟨raw,rawMem,rfl⟩ := List.mem_map.mp selectedMem
    have size : (AutotunePlan.tune total p).size≤Grid.limit :=
      max_le (by norm_num [Grid.limit]) (min_le_left _ _)
    exact le_trans (Nat.add_le_add_left size p.base) (head raw rawMem)

theorem inner_ordered (c : HostFP.Inner) :
    c.plans.Pairwise (fun p q => p.base+p.size≤q.base) := by
  unfold HostFP.Inner.plans
  split
  · exact List.Pairwise.nil
  · exact LaunchPlan.plans_ordered c.total

theorem prefix_ordered (c : HostFP.Cumulative) :
    c.plans.Pairwise (fun p q => p.base+p.size≤q.base) := by
  unfold HostFP.Cumulative.plans
  split
  · exact List.Pairwise.nil
  · exact tune_ordered c.total (LaunchPlan.plans c.maximum) (build_base_spacing _ _ _)

#axiomsClean inner_ordered
#axiomsClean prefix_ordered
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
