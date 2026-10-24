import bench.repairs.tritonbench_g.fast_layernorm.Proofs.ReturnLive

namespace VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation
noncomputable section
open scoped Classical
variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type} {lib : APIHost api neg inputs Metadata} {initial : State Store Metadata}

theorem optional_metadata {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat}
    (r : OptionalReturn lib shape device initial b) (source : Dense) (out : Storage.Tensor)
    (hs : b=some source) (ho : r.output=some out) :
    out.view.shape=shape.getD [] ∧ out.view.device=source.tensor.view.device ∧ out.view.shape.prod=source.count := by
  cases r with
  | none => contradiction
  | @some original tr =>
    have he : original=source := Option.some.inj hs
    subst source
    have heo : tr.output=out := by simpa [OptionalReturn.output] using ho
    subst out
    have h := transfer_metadata tr
    have hv := lib.viewed original (shape.getD []) initial tr.viewed tr.viewOK
    refine ⟨h.1,h.2.1,?_⟩
    rw [h.1,←hv.shape,←tr.viewed.countEq,hv.count]

theorem optional_presence {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat}
    (r : OptionalReturn lib shape device initial b) : r.output.isSome=b.isSome := by
  cases r <;> rfl

theorem db_metadata (r : Route lib c p initial) (source : Dense) (out : Storage.Tensor)
    (hs : p.db=some source) (ho : r.output.2.2.1=some out) :
    out.view.shape=c.biasShape.getD [] ∧ out.view.device=source.tensor.view.device ∧ out.view.shape.prod=source.count :=
  optional_metadata r.br source out hs ho

#axiomsClean db_metadata
#axiomsClean optional_presence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
