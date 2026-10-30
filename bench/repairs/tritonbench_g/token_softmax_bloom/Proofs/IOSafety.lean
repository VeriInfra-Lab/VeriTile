import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.KernelSafety
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.IOSchedule
import bench.repairs.tritonbench_g.token_softmax_bloom.Proofs.IOBinding

namespace VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenSoftmaxLlama.Strided
open SoftmaxIO
noncomputable section
set_option maxHeartbeats 2000000
variable {α β : Type} [Inhabited β]

theorem one (M : Algebra β) (p : Parameters) (positive : 0 < p.width)
    (bounds : RegionName → Nat) (base : Grid.Memory β) (pid : Grid.PID)
    (bounded : IOBinding.RowBounds p base pid bounds) (s : State β)
    (agree : Grid.Agree p s.mem base) : (invocation M p pid).Safe bounds s := by
  have hc : count p ((invocation M p pid).initial s) = count p (Grid.state base pid) :=
    Grid.count_congr p s.mem base pid agree
  have hs : startIndex p ((invocation M p pid).initial s) = startIndex p (Grid.state base pid) :=
    Grid.start_congr p s.mem base pid agree
  apply KernelSafety.source_safe M p positive _ bounds
  · exact bounded.1
  · exact bounded.2.1
  · intro i active
    rw [hc] at active
    change pid.2*p.inputHeadStride+ _ < _
    rw [hs]
    exact bounded.2.2.1 i active
  · intro i active
    rw [hc] at active
    change pid.2*p.outputHeadStride+ _ < _
    rw [hs]
    exact bounded.2.2.2 i active

/-- Read regions remain unchanged after each actual source launch, so every
successor uses the same metadata-derived bounds as the admitted input. -/
theorem schedule (M : Algebra β) (p : Parameters) (positive : 0 < p.width)
    (sep : Grid.Separate p) (work : List Grid.PID) (bounds : RegionName → Nat)
    (base : Grid.Memory β) (bounded : ∀ pid ∈ work, IOBinding.RowBounds p base pid bounds)
    (s : State β) (agree : Grid.Agree p s.mem base) :
    ScheduleSafe bounds (work.map (invocation M p)) s := by
  induction work generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨one M p positive bounds base pid (bounded pid (by simp)) s agree, ?_⟩
    obtain ⟨u, executed, memory⟩ := IOSchedule.one M p positive pid s
    intro t actual
    have same : u = t := Option.some.inj (executed.symm.trans actual)
    subst t
    apply ih (fun q hq => bounded q (List.mem_cons_of_mem _ hq)) u
    intro r hr
    rw [memory]
    exact (Grid.block_agrees M p pid s.mem sep r hr).trans (agree r hr)

theorem safe (M : Algebra β) (encode : α → β) (p : Parameters)
    (observe : Fin p.width → β → α) (positive : 0 < p.width)
    (sep : Grid.Separate p) (work : List Grid.PID)
    (vals : (io M encode p observe work).Inputs) (bounds : RegionName → Nat)
    (inside : (io M encode p observe work).InBounds bounds vals)
    (s : State β) (binding : (io M encode p observe work).InputBinding vals s) :
    ScheduleSafe bounds ((io M encode p observe work).schedule vals) s :=
  schedule M p positive sep work bounds s.mem
    (IOBinding.row_bounds M encode p observe work vals s binding bounds inside) s (fun _ _ => rfl)

#axiomsClean safe
end
end VeriTile.Bench.Repairs.TokenSoftmaxBloom.IOSafety
