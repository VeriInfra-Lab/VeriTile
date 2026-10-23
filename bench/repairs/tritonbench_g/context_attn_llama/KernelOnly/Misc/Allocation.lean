import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Tensor

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
noncomputable section
set_option maxHeartbeats 6000000

def roles : List RegionName :=
  [regions.query,regions.key,regions.value,regions.out,regions.start,regions.sequence,regions.prompt,regions.request,regions.pages]

def roleIndex (rg : RegionName) : Nat :=
  if rg=regions.query then 0 else if rg=regions.key then 1 else if rg=regions.value then 2
  else if rg=regions.out then 3 else if rg=regions.start then 4 else if rg=regions.sequence then 5
  else if rg=regions.prompt then 6 else if rg=regions.request then 7 else 8

/-- Derived nominal role layout. Physical read aliasing is supplied separately
by the common original heap; no physical read/read separation is assumed. -/
def allocation (r : Request) : FlatAlloc where
  flat := "flat"
  regions := roles
  base := fun rg => roleIndex rg*(queryCount r+cacheCount r+r.call.batches+pageCount r)
  extent := extent r

theorem extent_le_slot (r : Request) (rg : RegionName) :
    extent r rg≤queryCount r+cacheCount r+r.call.batches+pageCount r := by
  unfold extent
  split <;> first | omega | skip
  split <;> first | omega | skip
  split <;> first | omega | skip
  split <;> omega

theorem roleIndex_injective (rg rg' : RegionName) (hr : rg∈roles) (hr' : rg'∈roles)
    (same : roleIndex rg=roleIndex rg') : rg=rg' := by
  have locate (x : RegionName) (hx : x∈roles) : roles[roleIndex x]?=some x := by
    simp only [roles,List.mem_cons,List.not_mem_nil,or_false] at hx
    rcases hx with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl <;> rfl
  have found := congrArg (fun i => roles[i]?) same
  change roles[roleIndex rg]?=roles[roleIndex rg']? at found
  rw [locate rg hr,locate rg' hr'] at found
  exact Option.some.inj found

theorem allocation_disjoint (r : Request) : (allocation r).Disjoint := by
  intro rg hr rg' hr' different
  have indices : roleIndex rg≠roleIndex rg' := fun same => different (roleIndex_injective rg rg' hr hr' same)
  have leftBound := extent_le_slot r rg
  have rightBound := extent_le_slot r rg'
  rcases lt_or_gt_of_ne indices with before|after
  · left
    have scaled := Nat.mul_le_mul_right (queryCount r+cacheCount r+r.call.batches+pageCount r)
      (Nat.succ_le_of_lt before)
    simp only [Nat.succ_mul] at scaled
    change roleIndex rg*(queryCount r+cacheCount r+r.call.batches+pageCount r)+extent r rg≤_
    exact le_trans (Nat.add_le_add_left leftBound _) scaled
  · right
    have scaled := Nat.mul_le_mul_right (queryCount r+cacheCount r+r.call.batches+pageCount r)
      (Nat.succ_le_of_lt after)
    simp only [Nat.succ_mul] at scaled
    change roleIndex rg'*(queryCount r+cacheCount r+r.call.batches+pageCount r)+extent r rg'≤_
    exact le_trans (Nat.add_le_add_left rightBound _) scaled

theorem allocation_covered (r : Request) :
    ∀ rg,rg∉(allocation r).regions → (allocation r).extent rg=0 := by
  intro rg hr
  simp only [allocation,roles,List.mem_cons,List.not_mem_nil,or_false,not_or] at hr
  simp [allocation,extent,hr.1,hr.2.1,hr.2.2.1,hr.2.2.2.1,hr.2.2.2.2.1,
    hr.2.2.2.2.2.1,hr.2.2.2.2.2.2.1,hr.2.2.2.2.2.2.2.1,hr.2.2.2.2.2.2.2.2]

theorem layout_valid (r : Request) (heap : ByteMemory) (h : Contract r heap) :
    Host.Valid r.call (allocation r) (metadataState r heap) :=
  Checked.valid r.call r.dimensions (allocation r) (metadataState r heap)
    (indexValues r heap .start) (indexValues r heap .sequence) (indexValues r heap .prompt)
    (indexValues r heap .request) (indexValues r heap .pages)
    h.checks.2.1 h.checks.2.2.2 (metadata_observation r heap) (labels r)

theorem complete_grid (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (b : Nat) (hb : b<r.call.batches) : r.call.length b≤r.call.maxInput :=
  (h.checks.2.1.2.2.2.2.2.2.1 ⟨b,hb⟩).2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
