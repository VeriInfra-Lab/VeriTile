import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Tensor
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 3000000

def roles : List RegionName :=
  [regions.query,regions.key,regions.value,regions.out,regions.start,regions.sequence]

def roleIndex (rg : RegionName) : Nat :=
  if rg=regions.query then 0 else if rg=regions.key then 1 else if rg=regions.value then 2
  else if rg=regions.out then 3 else if rg=regions.start then 4 else 5

/-- A derived symbolic layout for the existing Real interpreter. Physical input
aliasing is retained by the common reader, independently of these role names. -/
def allocation (r : Request) : FlatAlloc where
  flat := "flat"
  regions := roles
  base := fun rg => roleIndex rg*(payloadCount r+r.call.batches)
  extent := extent r

theorem allocation_disjoint (r : Request) : (allocation r).Disjoint := by
  intro rg hr rg' hr' hne
  change rg∈roles at hr
  change rg'∈roles at hr'
  simp only [roles,List.mem_cons,List.not_mem_nil,or_false] at hr hr'
  rcases hr with rfl|rfl|rfl|rfl|rfl|rfl <;>
    rcases hr' with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp_all [allocation,roleIndex,extent,regions]
  all_goals omega

theorem allocation_covered (r : Request) :
    ∀ rg,rg∉(allocation r).regions → (allocation r).extent rg=0 := by
  intro rg hr
  simp only [allocation,roles,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,extent,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2.1,hr.2.2.2.2.2]

theorem layout_memory (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    Host.Memory r.call regions (allocation r) (metadataState r heap) :=
  Checked.memory r.call regions (allocation r) (metadataState r heap)
    (startValues r heap) (lengthValues r heap) h.checks.2.1 h.checks.2.2.2
    (metadata_observation r heap) labels

theorem complete_grid (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    r.call.completeGrid := Checked.complete r.call regions (extent r) h.checks.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
