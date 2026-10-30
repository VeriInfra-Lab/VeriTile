import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.BackwardCall
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.ForwardCall
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Readiness

namespace VeriTile.Bench.Repairs.LayerNormTriton.Readiness
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
open VeriTile.Bench.Repairs.FastLayernorm.Host
open VeriTile.Bench.Repairs.FastLayernorm.Readiness
open Storage
noncomputable section
abbrev TensorReady := @VeriTile.Bench.Repairs.FastLayernorm.Readiness.TensorReady
variable {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32} {inputs : List Storage.Tensor}
  {Metadata : Type} {platform : Compiler.Platform}

theorem allocation_ready (effect : Effect api neg inputs Metadata Dense) (s : State Store Metadata) (d : Dense)
    (allocation : AllocationResult effect s d) : TensorReady (effect.call.run s).1 d.tensor := by
  unfold TensorReady VeriTile.Bench.Repairs.FastLayernorm.Readiness.TensorReady
  rw [(effect.transition s).streams]
  exact allocation.ready

variable {entry : Compiler.Entry} {M N B K : Nat} {eps : Compiler.Shared.HostScalar}
  {request : Runtime.Request K inputs}
  {service : Runtime.Service platform api neg inputs Metadata entry M N B K eps request}
  {values : Compiler.Selected platform.decoder entry N B → Storage.World Store → ∀ k,Fin (request.buffers.output k).count → UInt32}
  {initial : State Store Metadata}

theorem route_output (r : Runtime.Route service values initial) (k : Fin K) :
    TensorReady r.final (request.buffers.output k).tensor := by
  have hs := (service.admitted r.code r.s1 r.schedule r.admitted).2.2.1
  have ht := (service.admit r.code).transition r.s1 |>.streams
  change r.s2.ready _ _ ∨ _
  right
  refine ⟨r.completed,k,rfl,?_⟩
  change r.s2.streams (request.buffers.output k).tensor.view.device=r.schedule.stream
  rw [(request.buffers.output k).tensor.owner]
  exact (congrArg r.s2.streams (request.buffers.owned k)).trans ((congrFun ht _).trans hs.symm)

theorem route_preserved (r : Runtime.Route service values initial) (t : Storage.Tensor) (ready : TensorReady initial t) :
    TensorReady r.final t := Or.inl (preserved r.preTransition t ready)

variable {lib : APIHost api neg inputs Metadata} {identity : Nat} {x w b : Storage.Tensor}
  {normalized : List Nat} {forwardService : Host.ForwardService platform api neg inputs Metadata}

theorem forward_return (f : Host.Call lib forwardService identity x w b normalized eps initial) :
    TensorReady f.final f.body.output.tensor := by
  have hy : TensorReady f.body.launch.final f.body.preparation.data.y.tensor := by
    cases h : f.body.launch with
    | skipped inactive =>
      let r := f.body.preparation
      have allocated := allocation_ready (lib.emptyLike r.xr.output) r.br.final r.y
        (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).1
      exact preserved ((lib.emptyStats x.view.device f.shape.rows).transition r.sR) _
        (preserved ((lib.emptyStats x.view.device f.shape.rows).transition r.sY) _ allocated)
    | ran active r => exact route_output r 2
  have hsave := preserved ((lib.save f.body.context).transition f.body.launch.final) _ hy
  have hv := view_ready lib f.body.preparation.data.y f.body.output x.view.shape f.body.sSave f.body.viewed hsave
  exact preserved ((lib.leave initial.owner).transition f.body.final) _ hv

variable {extra : BackwardHost.APIExtra api neg inputs Metadata} {c : SavedContext} {bias : Dense}
  {saved : BackwardHost.Saved c bias} {dy : Storage.Tensor}
  {backService : BackwardHost.Service platform api neg inputs Metadata}

theorem base_outputs_ready (r : BackwardHost.BaseRoute lib extra c bias dy initial) (k : Fin 3) :
    TensorReady r.final (BackwardHost.outputs r.data k).tensor := by
  fin_cases k
  · exact preserved ((extra.zerosLike bias).transition r.sW) _
      (preserved ((extra.zerosLike c.w).transition r.sX) _
        (allocation_ready (lib.emptyLike c.x) r.dyr.final r.dx (lib.emptyLikeResult c.x r.dyr.final r.dx r.dxOK).1))
  · exact preserved ((extra.zerosLike bias).transition r.sW) _
      (allocation_ready (extra.zerosLike c.w) r.sX r.dw (extra.zeroResult c.w r.sX r.dw r.dwOK).1)
  · exact allocation_ready (extra.zerosLike bias) r.sW r.db (extra.zeroResult bias r.sW r.db r.dbOK).1

theorem backward_return (r : BackwardHost.Call lib extra backService identity c bias saved dy initial) (k : Fin 3) :
    TensorReady r.final (r.body.returns.outputs k) := by
  have hp : ∀ j,TensorReady r.body.launch.final (BackwardHost.outputs r.body.preparation.data.base j).tensor := by
    intro j
    cases h : r.body.launch with
    | skipped hn => exact preserved r.body.preparation.scratch.transition _ (base_outputs_ready r.body.preparation.base j)
    | ran sc hs lr =>
      fin_cases j
      · exact route_preserved lr.last _ (route_output lr.first 0)
      · exact route_output lr.last 0
      · exact route_output lr.last 1
  apply preserved ((lib.leave r.accessState.owner).transition r.body.final)
  apply return_ready r.body.returns.reuse
  · intro t ht
    simp only [BackwardHost.Base.returnData,BackwardLayout.outputs,Option.toList_some,List.mem_append,
      List.mem_cons,List.not_mem_nil,or_false] at ht
    rcases ht with (hx|hw)|hb
    · subst t; exact hp 0
    · subst t; exact hp 1
    · subst t; exact hp 2
  · fin_cases k <;> simp [BackwardHost.ReturnRoute.outputs,BackwardHost.ReturnRoute.reuse,
      GradientReturn.returned,GradientReturn.Route.output,GradientReturn.OptionalReturn.output] <;> tauto

#axiomsClean forward_return
#axiomsClean backward_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Readiness
