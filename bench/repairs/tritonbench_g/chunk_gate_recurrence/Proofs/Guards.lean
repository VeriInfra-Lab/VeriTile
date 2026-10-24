import bench.repairs.tritonbench_g.chunk_gate_recurrence.Contract

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
open Machine

theorem launch_checked (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (nonempty : 0 < c.N) :
    (runtime.currentDevice=runtime.device ∧ c.heads < 2^31 ∧ c.KT ≤ 65535 ∧ c.VT ≤ 65535) ∧
    rangeCheck c runtime.seedStrides=Bool.true ∧
    pointerCheck c runtime.seedStrides seed runtime.abi.allocation=Bool.true := by
  simp only [check,Bool.and_eq_true,decide_eq_true_eq,Bool.or_eq_true] at accepted
  rcases accepted.2 with zero|launch
  · omega
  · exact ⟨launch.1.1,launch.1.2,launch.2⟩

theorem active_present (c : Shape) (seed : Bool) (r : Region) (p t j : Nat)
    (active : Active c seed r p t j) : present c seed r=Bool.true := by
  cases r <;> simp only [Active,present,Bool.or_eq_true,decide_eq_true_eq] at *
  all_goals first | rfl | tauto | omega

theorem backing_checked (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (r : Region) (pr : present c seed r=Bool.true)
    (nonempty : 0 < extent c runtime.seedStrides r) :
    (runtime.tensor r).base=(runtime.tensor r).allocationBase+4*(runtime.tensor r).storageOffset ∧
    (runtime.tensor r).base%4=0 ∧
    4*(runtime.tensor r).storageOffset+4*extent c runtime.seedStrides r ≤ (runtime.tensor r).allocationBytes ∧
    (runtime.tensor r).allocationBase+(runtime.tensor r).allocationBytes ≤ 2^64 := by
  have checked := accepted
  simp only [check,Bool.and_eq_true] at checked
  have checked := checked.1
  have member : r∈allRegions := by cases r <;> simp [allRegions]
  have fact := List.all_eq_true.mp checked r member
  simp only [pr,↓reduceIte,decide_eq_true_eq] at fact
  have result : extent c runtime.seedStrides r=0 ∨
      ((runtime.tensor r).base=(runtime.tensor r).allocationBase+4*(runtime.tensor r).storageOffset ∧
       (runtime.tensor r).base%4=0 ∧
       4*(runtime.tensor r).storageOffset+4*extent c runtime.seedStrides r ≤ (runtime.tensor r).allocationBytes ∧
       (runtime.tensor r).allocationBase+(runtime.tensor r).allocationBytes ≤ 2^64) := by tauto
  exact result.resolve_left (by omega)

theorem active_physical (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) (r : Region) (p t j : Nat)
    (active : Active c seed r p t j) :
    (runtime.abi.allocation r).Contains (offset c runtime.abi r p t j) ∧
    0 ≤ (runtime.abi.allocation r).dataPointer+4*offset c runtime.abi r p t j ∧
    (runtime.abi.allocation r).dataPointer+4*offset c runtime.abi r p t j+4 ≤ 2^64 := by
  have bound : 0 ≤ offset c runtime.abi r p t j ∧
      offset c runtime.abi r p t j < (extent c runtime.seedStrides r : Int) :=
    active_offset c seed runtime.abi r p t j active
  have extentPositive : 0 < extent c runtime.seedStrides r := by
    change 0 ≤ offset c runtime.abi r p t j ∧ offset c runtime.abi r p t j < (extent c runtime.seedStrides r : Int) at bound
    omega
  have checked := backing_checked c seed runtime accepted r (active_present c seed r p t j active) extentPositive
  change (((runtime.tensor r).allocationBase : Int) ≤ (runtime.tensor r).base+4*offset c runtime.abi r p t j ∧
      (runtime.tensor r).base+4*offset c runtime.abi r p t j+4 ≤
        (runtime.tensor r).allocationBase+(runtime.tensor r).allocationBytes) ∧
    0 ≤ (runtime.tensor r).base+4*offset c runtime.abi r p t j ∧
    (runtime.tensor r).base+4*offset c runtime.abi r p t j+4 ≤ 2^64
  omega

theorem machine_domain (c : Shape) (seed : Bool) (runtime : Runtime c seed)
    (accepted : check c seed runtime=Bool.true) : Domain c seed runtime.abi := by
  refine ⟨?_,?_,?_,?_,runtime.freshCaller,runtime.freshOutput⟩
  · intro h tk tv e j count hh hk hv he hj hc hn x member
    exact source_valid c runtime.seedStrides (launch_checked c seed runtime accepted hn).2.1
      h tk tv e j count hh hk hv he hj hc x member
  · intro h tk tv e j count hh hk hv he hj hc r x member
    have launch := launch_checked c seed runtime accepted (by omega : 0 < c.N)
    exact formed_safe c runtime.seedStrides seed runtime.abi.allocation launch.2.1 launch.2.2
      h tk tv e j count hh hk hv he hj hc r x member
  · intro r p t j active
    exact (active_physical c seed runtime accepted r p t j active).1
  · intro r p t j active
    exact (active_physical c seed runtime accepted r p t j active).2

#axiomsClean backing_checked
#axiomsClean active_physical
#axiomsClean machine_domain
end VeriTile.Bench.Repairs.ChunkGateRecurrence.Checked
