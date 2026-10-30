import bench.tritonbench_g.context_attn_mistral.Proofs.NativeAccess
import bench.tritonbench_g.context_attn_mistral.Proofs.NativeTags

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
variable {α : Type} [Inhabited α]

theorem implements (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence) :
    (programIO c o jobs).ImplementsOn (NativeArrays.metadata c) (target c o jobs) := by
  apply ArrayProgramIO.ImplementsOn.intro
  · exact io_safe c o jobs separate
  · intro a metadata s bound
    have snap:=binding_snapshot c o jobs a s bound
    have metadataState : KernelFootprint.Metadata c s := by
      rw [←NativeBindings.metadata_iff,snap]
      exact metadata
    obtain ⟨t,executed,memory⟩:=KernelReference.whole c o jobs s
    obtain ⟨u,ran,metadataState',outside⟩:=KernelFootprint.whole_frame c o jobs s metadataState separate
    have same:u=t:=Option.some.inj (ran.symm.trans executed)
    subst u
    refine ⟨t,?_,?_,?_⟩
    · exact (schedule_eq c o jobs s).trans executed
    · intro port i on
      fin_cases port
      change mapCell id (t.mem c.regions.out (c.offset jobs[i.1] i.2))=
        Cell.mk .real (NativeArrays.grid c o jobs a 5 (c.offset jobs[i.1] i.2))
      have mapped (v : Cell α) : mapCell id v=v := by
        cases v with | mk d value=>cases d <;> rfl
      rw [mapped,memory]
      have tagged:=NativeBindings.grid_real c o jobs s metadataState separate _
        (Or.inr ⟨jobs[i.1],List.getElem_mem _,i.2,on,rfl,rfl⟩)
      rw [tagged]
      congr 1
      have matched:=NativeBindings.grid_snapshot c o jobs s metadataState separate
      have readback:=congrFun (congrFun matched 5) (c.offset jobs[i.1] i.2)
      change ((KernelReference.grid c o jobs s).mem c.regions.out (c.offset jobs[i.1] i.2)).read .real=_ at readback
      rw [snap] at readback
      exact readback
    · intro rg off untouched
      apply outside rg off
      rintro ⟨idx,member,i,on,region,address⟩
      obtain ⟨j,hj,rfl⟩:=List.mem_iff_getElem.mp member
      have miss:=untouched.1 (0 : Fin 1) (⟨j,hj⟩,i) on
      exact miss.elim (fun h=>h region) (fun h=>h address)

/-- Original local instruction agreement transports all owners of the same
complete graph. It is not a Real=FP claim or a new transformation hypothesis. -/
theorem observes (c : Host.Call) (jobs : List (GridIndex c.grid))
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence)
    (actual reference : ScalarOps) (laws : NativeInstructions.ScalarLaws actual reference) :
    (nativeIO c jobs).ObservesOn (observer c actual) (NativeArrays.metadata c)
      (nativeTarget c jobs reference) := by
  have same:=laws.operations_eq actual reference
  subst reference
  exact ArrayProgramIO.ObservesOn.of_implementsOn
    (implements c (KernelNative.backend c.layout.D c.layout.window.toNat) jobs separate)
    (observer c actual)

#axiomsClean implements
#axiomsClean observes
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
