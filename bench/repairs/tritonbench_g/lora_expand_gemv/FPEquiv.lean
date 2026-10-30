import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Compiler

namespace VeriTile.Bench.Repairs.LoraExpandGemv.FPEquiv
open VeriTile.Triton FP.Structural FPValue Contract Direct.NativeCall
open FP.Equational (ReductionPlan)
noncomputable section
abbrev nativeReference := Direct.NativeCall.reference

theorem checked_fp (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State) (ok : Admitted t s (indices a before))
    (order : List Nat) (coverage : order.Perm (List.range (geometry t s).count)) :
    execute rest plans (geometry t s) order (entry (geometry t s) (indices a before))=
      some (finalMemory rest plans t s a before) ∧
    (∀ b n,b<t.indices.numel → n<width t.w →
      Native.eval ops schedule (bits a before) ((finalMemory rest plans t s a before "Y" (yoff (geometry t s) b n)).read .real)=
        nativeReference ops schedule plans t s a before b n) := by
  refine ⟨runtime_schedule rest plans t s a before ok order coverage,?_⟩
  intro b n hb hn
  rw [(complete rest plans t s a before ok).2.1 b n hb hn]
  rfl

/-- Exact bytes from source memory and six local scalar instruction families,
retaining supplied output identity, None return and the full complement frame. -/
theorem checked_caller_bytes (rest : Algebra Word) (ops : Native.ScalarOps) (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (binding : CompilerBinding ops schedule plans) (t : Inputs) (s : Scalars) (a : Binding t) (before : Raw.State)
    (ok : Admitted t s (indices a before)) :
    let result:=invoke rest binding.primitives binding.compilerSchedule binding.compilerPlans t s a before ok
    result.tensor=t.y ∧ result.returned=() ∧
    execution rest binding.compilerPlans t s a before=some (finalMemory rest binding.compilerPlans t s a before) ∧
    (∀ b n,b<t.indices.numel → n<width t.w → indices a before b≠-1 → ∀ byte : Fin 4,
      result.bytes a.output.window.owner ((a.output.window.key (yoff (geometry t s) b n)).address+byte.val)=
        Raw.unpack (nativeReference ops schedule plans t s a before b n) byte) ∧
    (∀ owner address,(∀ q,q<(geometry t s).count → indices a before (q/(geometry t s).N)≠-1 →
      ¬(key t s a q).covers owner address) → result.bytes owner address=before owner address) := by
  exact ⟨rfl,rfl,compiled rest ops schedule plans binding t s a before ok⟩

#axiomsClean checked_fp
#axiomsClean checked_caller_bytes
#specNonCircular nativeReference avoiding [execution,finalMemory,sourceStores,execute,Branches.source,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.FPEquiv
