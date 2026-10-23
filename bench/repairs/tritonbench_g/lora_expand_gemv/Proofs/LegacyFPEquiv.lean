import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Caller
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Numerical
open VeriTile.Triton FP.Structural FPValue Native Physical
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Complete source plus actual little-endian output stores, every grid schedule,
original logical strided indices, original supplied Y identity and None return.
Only the six scalar instruction families are deferred numerical equalities. -/
theorem relative_equiv (rest : Algebra Word) (actual reference : ScalarOps)
    (laws : ScalarLaws actual reference) (S : Schedule) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (memory : Bytes) (call : Host.Preparation g)
    (rawBinding : call.raw=indices memory) (schedule : List Nat)
    (all : schedule.Perm (List.range g.count)) :
    ∃ result,Host.invoke rest plans g call schedule=some result ∧ result.returned=none ∧
      result.outputIdentity=call.outputIdentity ∧ result.callerDevice=call.callerDevice ∧
      result.grid=(g.N,g.B) ∧
      (∀ b n,b<g.B → n<g.N → ∀ j : Fin 4,
        (both actual S (data memory) plans g schedule (entry g call.raw) memory).2 "Y"
          (4*yoff g b n+j.val)=
        encode (eval reference S (data memory) ((resultCell plans g call.raw b n).read .real)) j) ∧
      (∀ r byte,(r≠("Y" : RegionName) ∨
          ∀ b n,b<g.B → n<g.N → byte/4≠yoff g b n) →
        (both actual S (data memory) plans g schedule (entry g call.raw) memory).2 r byte=memory r byte) ∧
      (∀ byte,project actual S (data memory) result.memory byte=
        (both actual S (data memory) plans g schedule (entry g call.raw) memory).2 "Y" byte) := by
  cases operations_eq actual reference laws
  obtain ⟨final,he,hv,hf⟩ := whole_source rest plans g call.raw call.valid schedule all
  let result : Host.Result := ⟨final,none,call.outputIdentity,call.callerDevice,(g.N,g.B)⟩
  have hi : Host.invoke rest plans g call schedule=some result := by
    simp [Host.invoke,Host.host_prepare g call.raw call.valid call.contiguous call.four call.singletonStride,he,result]
  have hs := both_source rest actual S (data memory) plans g schedule
    (fun q hq => List.mem_range.mp (all.mem_iff.mp hq)) (entry g call.raw) memory
  have hm : final=(both actual S (data memory) plans g schedule (entry g call.raw) memory).1 :=
    Option.some.inj (he.symm.trans hs)
  have related : ∀ byte,project actual S (data memory) final byte=
      (both actual S (data memory) plans g schedule (entry g call.raw) memory).2 "Y" byte := by
    rw [hm]
    exact both_related actual S (data memory) plans g schedule (entry g call.raw) memory
      (entry_related actual S g memory call.raw)
  refine ⟨result,hi,rfl,rfl,rfl,rfl,?_,?_,related⟩
  · intro b n hb hn j
    rw [←related]
    have hd : (4*yoff g b n+j.val)/4=yoff g b n := by omega
    have hr : (4*yoff g b n+j.val)%4=j.val := by omega
    simp only [project,hd,hv b n hb hn]
    congr 1
    exact Fin.ext hr
  · intro r byte ho
    apply both_frame
    rcases ho with hr|ha
    · exact .inl hr
    · right
      intro q hq
      have hc := coordinates g q (List.mem_range.mp (all.mem_iff.mp hq))
      exact ha _ _ hc.2 hc.1

/-- The complete public numerical contract at actual caller allocation bytes. -/
theorem whole_equiv (rest : Algebra Word) (actual reference : ScalarOps)
    (laws : ScalarLaws actual reference) (S : Schedule) (plans : ∀ n, ReductionPlan n)
    (g : Geometry) (memory : Caller.Bytes) (call : Host.Preparation g)
    (rawBinding : call.raw=indices (Caller.bind g call.abi memory)) (schedule : List Nat)
    (all : schedule.Perm (List.range g.count)) :
    ∃ result,Host.invoke rest plans g call schedule=some result ∧ result.returned=none ∧
      result.outputIdentity=call.outputIdentity ∧ result.callerDevice=call.callerDevice ∧
      result.grid=(g.N,g.B) ∧
      (∀ b n,b<g.B → n<g.N → ∀ j : Fin 4,
        Caller.run actual S (data (Caller.bind g call.abi memory)) plans g call.abi schedule
          (entry g call.raw) memory call.abi.y.allocation
          (call.abi.y.byteBase+4*yoff g b n+j.val)=
        encode (eval reference S (data (Caller.bind g call.abi memory))
          ((resultCell plans g call.raw b n).read .real)) j) ∧
      (∀ r byte,(r≠call.abi.y.allocation ∨ byte<call.abi.y.byteBase ∨
          ∀ b n,b<g.B → n<g.N → (byte-call.abi.y.byteBase)/4≠yoff g b n) →
        Caller.run actual S (data (Caller.bind g call.abi memory)) plans g call.abi schedule
          (entry g call.raw) memory r byte=memory r byte) := by
  obtain ⟨result,he,hn,ho,hd,hgrid,hv,hf,hrel⟩ := relative_equiv rest actual reference laws S plans g
    (Caller.bind g call.abi memory) call rawBinding schedule all
  refine ⟨result,he,hn,ho,hd,hgrid,?_,?_⟩
  · intro b n hb hn j
    rw [Nat.add_assoc,Caller.related actual S (data (Caller.bind g call.abi memory)) plans g call.abi
      schedule (entry g call.raw) memory (Caller.bind g call.abi memory) (by intro byte; simp [Caller.bind])]
    exact hv b n hb hn j
  · intro r byte h
    apply Caller.frame
    rcases h with hr|hb|ha
    · exact .inl hr
    · exact .inr (.inl hb)
    · right; right
      intro q hq
      have hc := coordinates g q (List.mem_range.mp (all.mem_iff.mp hq))
      exact ha _ _ hc.2 hc.1

#axiomsClean relative_equiv
#axiomsClean whole_equiv
#specNonCircular ScalarLaws avoiding [Host.invoke,execute,Branches.source,FP.Structural.exec,value]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Numerical
