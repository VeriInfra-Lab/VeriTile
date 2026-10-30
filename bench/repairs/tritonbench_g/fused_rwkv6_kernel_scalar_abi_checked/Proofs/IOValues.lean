import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOExecution
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.LogicalInputBinding

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.ValueProof
open VeriTile Triton FP.Structural FusedRwkv6Kernel Source SourceCells
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem outside (M : Algebra α) (c : Shape) (hc : Sizes c) (scale : α)
    (xs : Inputs α c) (r : RegionName) (a : Nat) (outside : (programIO M c scale).Outside xs r a) :
    ¬Execution.OutputAt c r a := by
  rintro (⟨rfl,ha⟩|⟨final,rfl,ha⟩)
  · obtain ⟨p,hp,write⟩ := cover c hc (r:=FusedRwkv6Kernel.Region.o) (b:=a) (Or.inl ⟨rfl,ha⟩)
    have valid := (mem_programs c p).mp hp
    rcases write with ⟨n,hn,_,l,hl,hf,eq⟩|⟨_,bad,_⟩
    · let j : TileIndex [c.NK,c.Z,c.T,c.V] :=
        (⟨p.ik,valid.2.1⟩,⟨p.z,valid.1⟩,⟨visit c n,visit_lt c hn⟩,⟨valueIndex c p l,hf⟩,())
      have h := outside.1 (0 : Fin 2) j (Or.inl rfl)
      exact h.elim (fun h => h rfl) (fun h => h eq)
    · cases bad
  · obtain ⟨p,hp,write⟩ := cover c hc (r:=FusedRwkv6Kernel.Region.ht) (b:=a) (Or.inr ⟨final,rfl,ha⟩)
    have valid := (mem_programs c p).mp hp
    rcases write with ⟨_,_,bad,_⟩|⟨_,_,l,hl,j,hj,hf,he,eq⟩
    · cases bad
    · let idx : TileIndex [c.Z,c.K,c.V] :=
        (⟨p.z,valid.1⟩,⟨keyIndex c p j,he⟩,⟨valueIndex c p l,hf⟩,())
      have h := outside.1 (1 : Fin 2) idx (Or.inr final)
      exact h.elim (fun h => h rfl) (fun h => h eq)

theorem values (M : Algebra α) (c : Shape) (hc : Sizes c) (plan : Plans c)
    (reduction : ReductionMatches M c plan) (scale : α) (xs : Inputs α c) (s : State α)
    (binding : (programIO M c scale).InputBinding xs s) :
    ∃ t,runSchedule ((programIO M c scale).schedule xs) s=some t ∧
      (programIO M c scale).OutputBinding xs (formula (ops M) c plan scale xs) t ∧
      (programIO M c scale).Frame xs s t := by
  by_cases live : Launches c
  · obtain ⟨t,run,wordsEq,tags,frame⟩ := Execution.grid M c plan reduction scale (programs c)
      (fun p hp => (mem_programs c p).mp hp) s
    have agrees := Binding.initial M c hc scale xs s live binding
    have vals (r : FusedRwkv6Kernel.Region) (b : Nat) : words t.mem r b=
        if Output c r b then target (ops M) c plan (inputs c (words s.mem)) scale r b else words s.mem r b := by
      rw [wordsEq,complete_grid _ c hc]
    refine ⟨t,?_,?_,?_⟩
    · simpa only [programIO,if_pos live] using run
    · intro o j active
      change Fin 2 at o
      fin_cases o
      · let b := partialAddress c j.2.1.val j.1.val j.2.2.1.val j.2.2.2.1.val
        have output : Output c .o b := partial_output c _ _ _ _ j.2.1.isLt j.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt
        have typed := (tags .o b).mpr (Or.inr (cover c hc output))
        change RealCell (t.mem "o" b) at typed
        change mapCell id (t.mem "o" b)=Cell.mk .real (formula (ops M) c plan scale xs (0 : Fin 2) j)
        rw [realCell_read _ typed]
        change Cell.mk .real (words t.mem .o b)=_
        rw [vals,if_pos output]
        have dec := partial_decode c j.2.1.val j.1.val j.2.2.1.val j.2.2.2.1.val
          j.2.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt
        change Cell.mk .real (target (ops M) c plan (inputs c (words s.mem)) scale .o b)=_
        simp only [target,b,dec.1,dec.2.1,dec.2.2.1,dec.2.2.2,formula]
        congr 2
        exact Binding.partial_eq (ops M) c plan _ _ agrees scale _ _ _ _ j.2.1.isLt (visit_lt c j.2.2.1.isLt)
      · have final : c.final=Bool.true := active.resolve_left (by change ¬((1 : Nat)=0);decide)
        let b := stateAddress c j.1.val j.2.1.val j.2.2.1.val
        have output : Output c .ht b := terminal_output c _ _ _ final j.1.isLt j.2.1.isLt j.2.2.1.isLt
        have typed := (tags .ht b).mpr (Or.inr (cover c hc output))
        change RealCell (t.mem "ht" b) at typed
        change mapCell id (t.mem "ht" b)=Cell.mk .real (formula (ops M) c plan scale xs (1 : Fin 2) j)
        rw [realCell_read _ typed]
        change Cell.mk .real (words t.mem .ht b)=_
        rw [vals,if_pos output]
        have dec := state_decode c j.1.val j.2.1.val j.2.2.1.val j.2.1.isLt j.2.2.1.isLt
        simp only [target,b,dec.1,dec.2.1,dec.2.2,formula]
        congr 2
        exact Binding.state (ops M) c _ _ agrees _ _ _ c.T j.1.isLt le_rfl
    · intro r a h
      exact frame r a (outside M c hc scale xs r a h)
  · have ht : c.T=0 := by by_contra h;exact live (Or.inl h)
    have hf : c.final≠Bool.true := fun h => live (Or.inr h)
    refine ⟨s,?_,?_,?_⟩
    · simp [programIO,live,runSchedule]
    · intro o j active
      change Fin 2 at o
      fin_cases o
      · have h := j.2.2.1.isLt
        change _<c.T at h
        omega
      · exact False.elim (hf (active.resolve_left (by change ¬((1 : Nat)=0);decide)))
    · intro r a _
      rfl

#axiomsClean values
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.ValueProof
