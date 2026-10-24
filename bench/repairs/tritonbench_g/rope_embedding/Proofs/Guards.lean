import bench.repairs.tritonbench_g.rope_embedding.Proofs.HostContract
namespace VeriTile.Bench.Repairs.RopeEmbedding.Checked
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 3000000

theorem scheduled_bounds (r : Request) (l : Location r) (hl : l∈schedule r) :
    r.size≠0 ∧ r.half≠0 ∧ group l*4+4≤r.heads+3 ∧
    r.offset (row l) (head l) ((lane l).1.val+r.half)≤outputLast r ∧
    row l%r.sequence*r.half+(lane l).1.val≤tableLast r := by
  obtain ⟨hr,hg,hlo,hhi⟩ := (mem_schedule r l).mp hl
  have hh : head l<r.heads := lt_of_lt_of_le hhi (min_le_right _ _)
  have hj := (lane l).1.isLt
  have half : r.half≠0 := by intro h; simp [Request.block,h] at hj
  have dimension : 0<r.dimension := by unfold Request.half at half; omega
  have size : r.size≠0 := by unfold Request.size; exact Nat.ne_of_gt (Nat.mul_pos (Nat.mul_pos (by omega) (by omega)) dimension)
  have sequence : 0<r.sequence := by
    by_contra hn
    have hz : r.sequence=0 := by omega
    simp [Request.rows,hz] at hr
  have hm := Nat.mod_lt (row l) sequence
  have ho1 := Nat.mul_le_mul_right r.heads (show row l≤r.rows-1 by omega)
  have ho2 := Nat.mul_le_mul_right r.dimension
    (show row l*r.heads+head l≤(r.rows-1)*r.heads+(r.heads-1) by omega)
  have ht := Nat.mul_le_mul_right r.half (show row l%r.sequence≤r.sequence-1 by omega)
  refine ⟨size,half,?_,?_,?_⟩
  · unfold Request.groups at hg; omega
  · unfold Request.offset outputLast; omega
  · unfold tableLast; omega

theorem indexABI {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (clone squeeze : Bool) (accepted : check api embed r s call clone squeeze=Bool.true) :
    Native.IndexABI api embed r s call := by
  simp only [check,Bool.and_eq_true] at accepted
  have h := accepted.2
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at h
  obtain ⟨dimensions,offsets⟩ := h.1
  refine ⟨dimensions.1,dimensions.2.1,dimensions.2.2.1,dimensions.2.2.2.1,
    dimensions.2.2.2.2.1,dimensions.2.2.2.2.2.1,dimensions.2.2.2.2.2.2.1,
    dimensions.2.2.2.2.2.2.2,?_⟩
  intro l hl
  obtain ⟨size,half,groupBound,outputBound,tableBound⟩ := scheduled_bounds r l hl
  have indices := offsets ⟨size,half⟩
  have pointers := h.2 ⟨size,half⟩
  exact ⟨by omega,by omega,by omega,by omega,by omega,by omega⟩

theorem cache_admitted (r : Request) (groups block warps : Nat)
    (h : cacheCheck r groups block warps=Bool.true) :
    (groups,block,warps)=(r.groups,r.block,r.warps) := by simpa [cacheCheck] using h

theorem path_admitted {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (clone squeeze : Bool) (accepted : check api embed r s call clone squeeze=Bool.true) :
    call.cloneInput=clone ∧ call.squeezed=squeeze := by
  simp only [check,metadataCheck,Bool.and_eq_true,decide_eq_true_eq] at accepted
  tauto

theorem forward_fresh {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (embed : β→α) (r : Request) (s : State α) (call : Native.Preparation api embed r s)
    (accepted : check api embed r s call Bool.true Bool.true=Bool.true) (b : Nat) (hb : call.caller b) :
    b<call.output.base ∨ call.output.base+4*r.size≤b := by
  have clone := (path_admitted api embed r s call Bool.true Bool.true accepted).1
  have reuse : call.reused=Bool.false := by simp [call.reuseCondition,clone]
  exact call.fresh reuse b hb

#axiomsClean scheduled_bounds
#axiomsClean indexABI
#axiomsClean cache_admitted
#specNonCircular check avoiding [Native.entry,Memory.run,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RopeEmbedding.Checked
