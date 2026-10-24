import bench.repairs.tritonbench_g.fast_layernorm.Proofs.GradientValues

namespace VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host BackwardPreparation
noncomputable section
open scoped Classical
variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {Metadata : Type} {lib : APIHost api neg inputs Metadata} {initial : State Store Metadata}

theorem optional_live {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat}
    (r : OptionalReturn lib shape device initial b) (out : Storage.Tensor) (h : r.output=some out) :
    r.final.live out := by
  cases r with
  | none => simp [OptionalReturn.output] at h
  | some tr =>
    have he : tr.output=out := by simpa [OptionalReturn.output] using h
    subst out
    exact (transfer_metadata tr).2.2

def returned (out : Gradients) : List Storage.Tensor := [out.1,out.2.1]++out.2.2.1.toList

theorem return_live (r : Route lib c p initial) (t : Storage.Tensor) (member : t∈returned r.output) : r.final.live t := by
  have hx := (lib.viewed p.dx c.inputShape initial r.dx r.dxOK).present
  have hw := (transfer_metadata r.wr).2.2
  simp only [returned,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (hx'|hw')|hb
  · subst t
    exact r.afterDX.live _ hx
  · subst t
    exact r.br.transition.live _ hw
  · exact optional_live r.br t (by simpa using hb)

#axiomsClean return_live
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.GradientReturn
