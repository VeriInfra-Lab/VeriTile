import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.FormulaAgreement
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Proofs.RegionalGrid

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

theorem source_safe (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity)
    (s : BlockState) (valid : Host.Valid c (capacityAlloc capacity) s)
    (bounds : RegionBounds) (backed : (programIO c capacity).InBounds bounds s values)
    (idx : GridIndex c.grid) :
    Kernel.TraceSafe bounds (source c).kernel.toAlgKernel (s.withGridIndex idx) := by
  have input (port : Fin 8) : capacity (inputRegion c port)≤bounds (inputRegion c port) := by
    by_contra bad
    have h : bounds (inputRegion c port)<capacity (inputRegion c port) := by omega
    have contra:=backed.1 port ⟨bounds (inputRegion c port),h⟩ trivial
    exact (Nat.lt_irrefl _) contra
  have old:=Host.program_bounds c (capacityAlloc capacity) s valid idx
  have grow {address : Int} {region : RegionName}
      (old : AddressBounds.inside address (capacity region)) (size : capacity region≤bounds region) :
      AddressBounds.inside address (bounds region) := by
    rcases old with ⟨nonnegative,inside⟩
    exact ⟨nonnegative,lt_of_lt_of_le inside (Int.ofNat_le.mpr size)⟩
  apply WholeSafety.complete c.layout c.regions .fp32 Bool.false (Host.scale c.layout.D)
    (s.withGridIndex idx) (by rw [c.blockN]; exact Host.bm_positive c) bounds
  refine {
    start:=lt_of_lt_of_le old.start (input 3)
    prompt:=lt_of_lt_of_le old.prompt (input 5)
    sequence:=lt_of_lt_of_le old.sequence (input 4)
    request:=lt_of_lt_of_le old.request (input 6)
    pages:=fun step hs j on=>grow (old.pages step hs j on) (input 7)
    query:=fun i d on=>grow (old.query i d on) (input 0)
    key:=fun step hs j d on=>grow (old.key step hs j d on) (input 1)
    value:=fun step hs j d on=>grow (old.value step hs j d on) (input 2)
    output:=?_ }
  intro i d on
  have active : c.active idx (i,d,()) :=
    (Host.active_eq c (capacityAlloc capacity) s valid idx (i,d,())).mp on
  have write:=backed.2.1 (0 : Fin 1) (lanes c (idx,(i,d,())))
    (by simpa only [programIO,Equiv.symm_apply_apply] using active)
  change c.offset ((lanes c).symm (lanes c (idx,(i,d,())))).1
    ((lanes c).symm (lanes c (idx,(i,d,())))).2 < bounds c.regions.out at write
  simp only [Equiv.symm_apply_apply] at write
  rw [←Host.offset_eq c (capacityAlloc capacity) s valid idx (i,d,())] at write
  have nonnegative:=(old.output i d on).1
  change (AddressBounds.outputAddress c.layout (Program.launch (s.withGridIndex idx))
    (Program.inputs (s.withGridIndex idx) c.regions) i.val d.val).toNat < _ at write
  exact ⟨nonnegative,by omega⟩

theorem implements (c : Host.Call) (capacity : RegionBounds) (geometry : Geometry c capacity) :
    (programIO c capacity).ImplementsOn (Metadata c capacity) (formula c capacity) := by
  intro values metadata s bound
  have valid:=host_valid c capacity geometry values metadata s bound
  obtain ⟨t,execution,outputs,outside⟩:=whole c (capacityAlloc capacity) s valid
  let trace : AlgorithmExecution [.ordinary (source c) c.grid] s t:=
    .cons execution (.nil t)
  refine ⟨t,trace,?_,?_,?_⟩
  · intro bounds backed
    exact ⟨fun idx=>source_safe c capacity values s valid bounds backed idx,trivial⟩
  · intro o j active
    let k:=(lanes c).symm j
    change t.readMem c.regions.out (c.offset k.1 k.2)=_
    rw [outputs k.1 k.2 active]
    exact formula_agreement c capacity (Program.inputs s c.regions) (logicalInputs c capacity values)
      (input_agreement c capacity values s bound) (c.launchAt k.1)
      (Host.program_bounds c (capacityAlloc capacity) s valid k.1)
      (Host.scale c.layout.D) k.2.1 k.2.2.1
  · intro r off output scratch
    apply outside r off
    rintro ⟨idx,region,i,active,address⟩
    have h:=output (0 : Fin 1) (lanes c (idx,i))
      (by simpa only [programIO,Equiv.symm_apply_apply] using active)
    change r≠c.regions.out ∨ off≠c.offset ((lanes c).symm (lanes c (idx,i))).1
      ((lanes c).symm (lanes c (idx,i))).2 at h
    simp only [Equiv.symm_apply_apply] at h
    exact h.elim (fun ne=>ne region) (fun ne=>ne address.symm)

#axiomsClean source_safe
#axiomsClean implements
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.RealInterface
