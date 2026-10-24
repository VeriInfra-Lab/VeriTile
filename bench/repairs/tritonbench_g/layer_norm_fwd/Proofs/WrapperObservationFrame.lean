import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperPreparedObservation

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
open VeriTile.Triton FP.Structural
open Wrapper
noncomputable section
set_option maxHeartbeats 3000000
variable {S α Descriptor : Type}

theorem readSecondMatrix_extends (store : TensorStore S Descriptor (Cell α)) (s t : S)
    (he : store.Extends s t) (M N : Nat) (value : Tensor) (hp : store.present s value) :
    readSecondMatrix store t M N value=readSecondMatrix store s M N value :=
  funext fun i => funext fun j => (he value hp).2.2 [M+i.val,j.val]

theorem preparedObserve_extends (store : TensorStore S Descriptor (Cell α)) (s t : S)
    (he : store.Extends s t) (p : Prepared) (hp : PreparedPresent store p s) (K : Nat) :
    preparedObserve store t p K=preparedObserve store s p K := by
  apply TypedSource.Result.ext
  · exact readMatrix_extends store s t he _ _ _ hp.y
  · exact optional_read_extends store s t he p.outputs.y1 hp.y1
      (fun s tensor => readMatrix store s p.validated.M p.validated.N tensor)
      (fun tensor hp => readMatrix_extends store s t he _ _ tensor hp)
  · exact optional_read_extends store s t he p.outputs.mean hp.mean
      (fun s tensor => readVector store s p.validated.M tensor)
      (fun tensor hp => readVector_extends store s t he _ tensor hp)
  · exact readVector_extends store s t he _ _ hp.rstd
  · exact readMatrix_extends store s t he _ _ _ hp.fallback
  · exact optional_read_extends store s t he p.outputs.seeds hp.seeds
      (fun s tensor => readVector store s K tensor)
      (fun tensor hp => readVector_extends store s t he _ tensor hp)
  · exact optional_read_extends store s t he p.outputs.mask hp.mask
      (fun s tensor => readMatrix store s p.validated.M p.validated.N tensor)
      (fun tensor hp => readMatrix_extends store s t he _ _ tensor hp)
  · dsimp only [preparedObserve]
    split
    · exact optional_read_extends store s t he p.outputs.mask hp.mask
        (fun s tensor => readSecondMatrix store s p.validated.M p.validated.N tensor)
        (fun tensor hp => readSecondMatrix_extends store s t he _ _ tensor hp)
    · rfl

theorem optional_present_host {Host : Type}
    (store : TensorStore (HostState α Host) WrapperNative.Descriptor (Cell α)) (hh : HostDescriptors store)
    (s t : HostState α Host) (he : s.host=t.host) (value : Option Tensor)
    (hp : OptionPresent store s value) : OptionPresent store t value := by
  cases value with
  | none => trivial
  | some value => exact hh.present s t he value hp

theorem PreparedPresent.host_transfer {Host : Type}
    (store : TensorStore (HostState α Host) WrapperNative.Descriptor (Cell α)) (hh : HostDescriptors store)
    (s t : HostState α Host) (he : s.host=t.host) (p : Prepared)
    (hp : PreparedPresent store p s) : PreparedPresent store p t :=
  ⟨hh.present s t he _ hp.y,optional_present_host store hh s t he _ hp.y1,
    optional_present_host store hh s t he _ hp.mean,hh.present s t he _ hp.rstd,
    optional_present_host store hh s t he _ hp.residualOut,hh.present s t he _ hp.original,
    optional_present_host store hh s t he _ hp.seeds,optional_present_host store hh s t he _ hp.mask⟩

#axiomsClean readSecondMatrix_extends
#axiomsClean preparedObserve_extends
#axiomsClean optional_present_host
#axiomsClean PreparedPresent.host_transfer
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperNative
