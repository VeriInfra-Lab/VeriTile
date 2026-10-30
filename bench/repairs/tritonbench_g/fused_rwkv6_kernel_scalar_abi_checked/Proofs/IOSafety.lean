import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.SafetyLoop
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOExecution

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.SafetyProof
open VeriTile Triton FP.Structural FusedRwkv6Kernel Source
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem source_bounds (M : Algebra α) (c : Shape) (hc : Sizes c) (scale : α)
    (xs : Inputs α c) (bounds : RegionName → Nat) (bounded : (programIO M c scale).InBounds bounds xs)
    (live : Launches c) (p : FusedRwkv6Kernel.Program) (valid : Valid c p) (s : State α) :
    SourceSafety.Bounds c ((invocation M c scale p).initial s) bounds := by
  constructor
  · intro j hj
    change p.ik*c.BK+j.val<c.K at hj
    have h := bounded.2.1 (4 : Fin 6) (⟨p.z%c.H,Nat.mod_lt _ hc.H⟩,⟨p.ik*c.BK+j.val,hj⟩,())
      ⟨live,Or.inl (by change (4 : Nat)≠5; decide)⟩
    change (p.z%c.H)*c.K+j.val+p.ik*c.BK < bounds "u"
    simpa [programIO,inputRead,inputPorts,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro hi l j hl hj
    change p.iv*c.BV+l.val<c.V at hl
    change p.ik*c.BK+j.val<c.K at hj
    have h := bounded.2.1 (5 : Fin 6)
      (⟨p.z,valid.1⟩,⟨p.ik*c.BK+j.val,hj⟩,⟨p.iv*c.BV+l.val,hl⟩,()) ⟨live,Or.inr hi⟩
    change p.z*c.K*c.V+(p.ik*c.BK+j.val)*c.V+(p.iv*c.BV+l.val) < bounds "h0"
    simpa [programIO,inputRead,inputPorts,stateAddress,Nat.add_mul,Nat.mul_assoc] using h
  · intro n hn j hj r hr
    change p.ik*c.BK+j.val<c.K at hj
    change pointer c (p.z*c.keyPitch+p.ik*c.BK+j.val) c.K n < bounds r
    rw [pointer_live c _ _ n hn]
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl
    all_goals
      first
      | have h := bounded.2.1 (0 : Fin 6)
          (⟨p.z,valid.1⟩,⟨visit c n,visit_lt c hn⟩,⟨p.ik*c.BK+j.val,hj⟩,())
          ⟨live,Or.inl (by change (0 : Nat)≠5; decide)⟩
        simpa [programIO,inputRead,inputPorts,inputKey,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
      | have h := bounded.2.1 (1 : Fin 6)
          (⟨p.z,valid.1⟩,⟨visit c n,visit_lt c hn⟩,⟨p.ik*c.BK+j.val,hj⟩,())
          ⟨live,Or.inl (by change (1 : Nat)≠5; decide)⟩
        simpa [programIO,inputRead,inputPorts,inputKey,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
      | have h := bounded.2.1 (3 : Fin 6)
          (⟨p.z,valid.1⟩,⟨visit c n,visit_lt c hn⟩,⟨p.ik*c.BK+j.val,hj⟩,())
          ⟨live,Or.inl (by change (3 : Nat)≠5; decide)⟩
        simpa [programIO,inputRead,inputPorts,inputKey,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro n hn l hl
    change p.iv*c.BV+l.val<c.V at hl
    have h := bounded.2.1 (2 : Fin 6)
      (⟨p.z,valid.1⟩,⟨visit c n,visit_lt c hn⟩,⟨p.iv*c.BV+l.val,hl⟩,())
      ⟨live,Or.inl (by change (2 : Nat)≠5; decide)⟩
    change pointer c (p.z*c.valuePitch+p.iv*c.BV+l.val) c.V n < bounds "v"
    rw [pointer_live c _ _ n hn]
    simpa [programIO,inputRead,inputPorts,inputValue,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro n hn l hl
    change p.iv*c.BV+l.val<c.V at hl
    have h := bounded.2.2.1 (0 : Fin 2)
      (⟨p.ik,valid.2.1⟩,⟨p.z,valid.1⟩,⟨visit c n,visit_lt c hn⟩,⟨p.iv*c.BV+l.val,hl⟩,())
      (Or.inl rfl)
    change pointer c ((p.z+p.ik*c.B*c.H)*(c.T*c.V)+p.iv*c.BV+l.val) c.V n < bounds "o"
    rw [pointer_live c _ _ n hn]
    simpa [programIO,outputWrite,outputPorts,partialAddress,Shape.Z,
      Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using h
  · intro hf l j hl hj
    change p.iv*c.BV+l.val<c.V at hl
    change p.ik*c.BK+j.val<c.K at hj
    have h := bounded.2.2.1 (1 : Fin 2)
      (⟨p.z,valid.1⟩,⟨p.ik*c.BK+j.val,hj⟩,⟨p.iv*c.BV+l.val,hl⟩,()) (Or.inr hf)
    change p.z*c.K*c.V+(p.ik*c.BK+j.val)*c.V+(p.iv*c.BV+l.val) < bounds "ht"
    simpa [programIO,outputWrite,outputPorts,stateAddress,Nat.add_mul,Nat.mul_assoc] using h

theorem invocation_safe (M : Algebra α) (c : Shape) (hc : Sizes c) (scale : α)
    (xs : Inputs α c) (bounds : RegionName → Nat) (bounded : (programIO M c scale).InBounds bounds xs)
    (live : Launches c) (p : FusedRwkv6Kernel.Program) (valid : Valid c p) (s : State α) :
    (invocation M c scale p).Safe bounds s :=
  SourceSafety.source_safe M c ((invocation M c scale p).initial s) scale
    (Execution.invocation_scalar M c scale p s) bounds
    (source_bounds M c hc scale xs bounds bounded live p valid s)

theorem schedule_safe (M : Algebra α) (c : Shape) (hc : Sizes c) (scale : α)
    (xs : Inputs α c) (bounds : RegionName → Nat) (bounded : (programIO M c scale).InBounds bounds xs)
    (live : Launches c) (order : List FusedRwkv6Kernel.Program)
    (valid : ∀ p∈order,Valid c p) (s : State α) :
    ScheduleSafe bounds (order.map (invocation M c scale)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons p rest ih =>
    exact ⟨invocation_safe M c hc scale xs bounds bounded live p (valid p (by simp)) s,
      fun next _ => ih (fun q hq => valid q (List.mem_cons_of_mem _ hq)) next⟩

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.SafetyProof
