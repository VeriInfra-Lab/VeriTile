import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulForward

namespace VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Wrapper
open VeriTile.Bench.Repairs.LayerNormFwd.Wrapper
noncomputable section
open scoped Classical
variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {metadata : MetadataAPI api neg inputs Metadata}
  {service : Runtime.Service platform api neg inputs Metadata} {identity : Nat}
  {module : Module} {x : Storage.Tensor} {initial : State Store Metadata}

theorem public_route (out : Dense)
    (ok : ((publicForward lib metadata service identity module x).run initial).result=.ok out) :
    module.affine=Bool.true ∧
    ∃ r : SuccessfulCall lib metadata service identity x module.weight module.bias module.selectedEpsilon initial,
      r.body.output=out ∧ ((publicForward lib metadata service identity module x).run initial).state=r.final := by
  obtain ⟨unit,hcheck,hforward,hstate,_⟩ := bind_success _ _ _ _ ok
  have ha : module.affine=Bool.true := by
    by_contra h
    simp [check,h] at hcheck
  obtain ⟨r,he,hs⟩ := forward_route out hforward
  exact ⟨ha,r,he,hstate.trans hs⟩

theorem public_frame (lib : APIHost api neg inputs Metadata) (metadata : MetadataAPI api neg inputs Metadata)
    (service : Runtime.Service platform api neg inputs Metadata) (identity : Nat) (module : Module)
    (x : Storage.Tensor) (s : State Store Metadata) (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api ((publicForward lib metadata service identity module x).run s).state.memory key=
      Storage.observe api s.memory key := (publicForward lib metadata service identity module x).frame s key hk

theorem epsilon_priority (module : Module) (value : Compiler.Shared.HostScalar)
    (present : module.varianceEpsilon=some value) : module.selectedEpsilon=value := by
  simp [Module.selectedEpsilon,present]

#axiomsClean public_route
#axiomsClean public_frame
#axiomsClean epsilon_priority
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.ForwardCall
