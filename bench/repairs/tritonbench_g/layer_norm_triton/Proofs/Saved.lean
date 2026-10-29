import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardCall
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Backward

namespace VeriTile.Bench.Repairs.LayerNormTriton.Host
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open Storage Preparation Layout
noncomputable section
open scoped Classical
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}
  {lib : APIHost api neg inputs Metadata} {service : ForwardService platform api neg inputs Metadata}
  {identity : Nat} {x w b : Storage.Tensor} {normalized : List Nat} {eps : Compiler.Shared.HostScalar}
  {initial : State Store Metadata}

theorem saved_layout (r : Call lib service identity x w b normalized eps initial) :
    ∃ bias,BackwardHost.Saved r.body.context bias := by
  cases hbr : r.body.preparation.br with
  | some br =>
    have hs := Geometry.shape_product r.shape
    have hc := Layout.route_counts r.body.preparation hs.1
    have hb : r.body.preparation.data.bias=some br.output := by
      simp [Preparation.Route.data,hbr,OptionalRoute.output]
    have bx : br.output.owner=r.body.preparation.data.x.owner :=
      Layout.route_input_owners r.body.preparation br.output (by simp [Layout.reads,hb])
    have wx : r.body.preparation.data.w.owner=r.body.preparation.data.x.owner :=
      Layout.route_input_owners r.body.preparation _ (by simp [Layout.reads])
    refine ⟨br.output,⟨hb,hc.1,hc.2.2.2.2.trans hs.2.1,
      (materialize_metadata br).2.1.trans hs.2.2,hc.2.2.1,hc.2.2.2.1,
      (materialize_metadata r.body.preparation.xr).1,wx,bx,?_,?_,?_,?_,r.settings⟩⟩
    · exact hs.1.trans hc.1.symm
    · exact hc.2.2.2.2.symm
    · exact ⟨b.view.shape,rfl,(materialize_metadata br).2.1.symm⟩
    · exact ⟨b.view.device,rfl⟩

theorem saved_live (r : Call lib service identity x w b normalized eps initial)
    (t : Dense) (ht : t∈[r.body.context.x,r.body.context.w,r.body.context.r,r.body.context.mu]++r.body.context.bias.toList) :
    r.final.live t.tensor := by
  have hprep : r.body.preparation.final.live t.tensor := by
    simp only [BodyRoute.context,savedContext,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with (hx|hw|hr|hm)|hb
    · subst t; exact (Layout.route_present r.body.preparation).1 _ (by simp [Layout.reads])
    · subst t; exact (Layout.route_present r.body.preparation).1 _ (by simp [Layout.reads])
    · subst t; exact (Layout.route_present r.body.preparation).2 0
    · subst t; exact (Layout.route_present r.body.preparation).2 1
    · exact (Layout.route_present r.body.preparation).1 _ (by simp [Layout.reads,hb])
  have hl := launch_live r.body.launch t.tensor hprep
  exact (lib.leave initial.owner).transition r.body.final |>.live _ (r.body.tail.live _ hl)

#axiomsClean saved_layout
#axiomsClean saved_live
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Host
