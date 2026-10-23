import bench.repairs.tritonbench_g.fast_layernorm.Proofs.SuccessfulForward
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardPost

/-! Output readiness follows the actual completed launches and ordinary view,
view and context transitions. Affine gradients are ready on their original
device streams; the candidate does not transfer tensors between devices. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Readiness
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host
noncomputable section
open scoped Classical

def TensorReady (s : State Store Metadata) (t : Storage.Tensor) : Prop :=
  s.ready t.backing (s.streams t.view.device)

theorem preserved {api : Memory.API α Store Byte} {neg : α → α}
    {inputs : List Storage.Tensor} {s u : State Store Metadata}
    (tr : Transition api neg inputs s u) (t : Storage.Tensor) (h : TensorReady s t) : TensorReady u t := by
  unfold TensorReady
  rw [tr.streams]
  exact tr.ready _ _ h

theorem view_ready {api : Memory.API α Store Byte} {neg : α → α}
    {inputs : List Storage.Tensor} (lib : APIHost api neg inputs Metadata)
    (t u : Dense) (shape : List Nat) (s : State Store Metadata)
    (ok : ((lib.view t shape).call.run s).2=.ok u) (h : TensorReady s t.tensor) :
    TensorReady ((lib.view t shape).call.run s).1 u.tensor := by
  have hv := lib.viewed t shape s u ok
  have hd : u.tensor.view.device=t.tensor.view.device := by
    rw [u.tensor.owner,t.tensor.owner,hv.backing]
  unfold TensorReady
  rw [hv.backing,hd,(lib.view t shape).transition s |>.streams]
  exact (lib.view t shape).transition s |>.ready _ _ h

theorem transfer_ready {api : Memory.API α Store Byte} {neg : α → α}
    {inputs : List Storage.Tensor} {lib : APIHost api neg inputs Metadata}
    {t : Dense} {shape : List Nat} {device : Nat} {s : State Store Metadata}
    (r : GradientReturn.TransferRoute lib t shape device s) (h : TensorReady s t.tensor) :
    TensorReady r.final r.output := by
  exact view_ready lib t r.viewed shape s r.viewOK h

theorem optional_ready {api : Memory.API α Store Byte} {neg : α → α}
    {inputs : List Storage.Tensor} {lib : APIHost api neg inputs Metadata}
    {b : Option Dense} {shape : Option (List Nat)} {device : Option Nat} {s : State Store Metadata}
    (r : GradientReturn.OptionalReturn lib shape device s b)
    (h : ∀ t,b=some t → TensorReady s t.tensor) (out : Storage.Tensor)
    (ho : r.output=some out) : TensorReady r.final out := by
  cases r with
  | none => simp [GradientReturn.OptionalReturn.output] at ho
  | some tr =>
    have he : tr.output=out := by simpa [GradientReturn.OptionalReturn.output] using ho
    subst out
    exact transfer_ready tr (h _ rfl)

theorem return_ready {api : Memory.API α Store Byte} {neg : α → α}
    {inputs : List Storage.Tensor} {lib : APIHost api neg inputs Metadata}
    {c : SavedContext} {p : BackwardPreparation.Data} {s : State Store Metadata}
    (r : GradientReturn.Route lib c p s)
    (h : ∀ t,t∈BackwardLayout.outputs p → TensorReady s t.tensor)
    (out : Storage.Tensor) (ho : out∈GradientReturn.returned r.output) : TensorReady r.final out := by
  have hx := view_ready lib p.dx r.dx c.inputShape s r.dxOK (h _ (by simp [BackwardLayout.outputs]))
  have hw := transfer_ready r.wr (preserved ((lib.view p.dx c.inputShape).transition s)
    p.dw.tensor (h _ (by simp [BackwardLayout.outputs])))
  simp only [GradientReturn.returned,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ho
  rcases ho with (hx'|hw')|hb
  · subst out
    exact preserved r.afterDX _ hx
  · subst out
    exact preserved r.br.transition _ hw
  · apply optional_ready r.br _ out (by simpa using hb)
    intro t ht
    apply preserved (((lib.view p.dx c.inputShape).transition s).then r.wr.transition)
    exact h t (by simp [BackwardLayout.outputs,ht])

variable {Store Metadata : Type} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {platform : Compiler.Platform}

theorem forward_output {service : Runtime.Service platform api neg inputs Metadata}
    {p : Preparation.Data} {rows N B : Nat} {layout : Layout.Forward p rows N inputs}
    {settings : Compiler.settings N=.ok (B,Compiler.warps B)} {eps : Compiler.Shared.HostScalar}
    {s : State Store Metadata} (r : Runtime.Route service p layout settings eps s) (k : Fin 3) :
    TensorReady r.final (Layout.outputs p k).tensor := by
  have hs := (service.admitted p layout r.code r.s1 r.schedule r.admitted).2.2.1
  have ht := (service.admit p layout r.code).transition r.s1 |>.streams
  have ho := layout.outputsOwned k
  change r.s2.ready _ _ ∨ _
  right
  refine ⟨r.completed,k,rfl,?_⟩
  change r.s2.streams (Layout.outputs p k).tensor.view.device=r.schedule.stream
  rw [(Layout.outputs p k).tensor.owner]
  exact (congrArg r.s2.streams ho).trans ((congrFun ht _).trans hs.symm)

theorem backward_output {service : BackwardRuntime.Service platform api neg inputs Metadata}
    {c : SavedContext} {p : BackwardPreparation.Data} {layout : BackwardLayout.Layout c p inputs}
    {kind : BackwardRuntime.Kind} {s : State Store Metadata}
    (r : BackwardRuntime.Route service c p layout kind s) (k : Fin (kind.count p)) :
    TensorReady r.final (kind.output p k).tensor := by
  have hs := (service.admitted c p layout kind r.code r.s1 r.schedule r.admitted).2.2.1
  have ht := (service.admit c p layout kind r.code).transition r.s1 |>.streams
  have ho := layout.owners _ (BackwardRuntime.output_mem kind p k)
  change r.s2.ready _ _ ∨ _
  right
  refine ⟨r.completed,k,rfl,?_⟩
  change r.s2.streams (kind.output p k).tensor.view.device=r.schedule.stream
  rw [(kind.output p k).tensor.owner]
  exact (congrArg r.s2.streams ho).trans ((congrFun ht _).trans hs.symm)

theorem backward_preserved {service : BackwardRuntime.Service platform api neg inputs Metadata}
    {c : SavedContext} {p : BackwardPreparation.Data} {layout : BackwardLayout.Layout c p inputs}
    {kind : BackwardRuntime.Kind} {s : State Store Metadata}
    (r : BackwardRuntime.Route service c p layout kind s) (t : Storage.Tensor) (h : TensorReady s t) :
    TensorReady r.final t := by
  exact Or.inl (preserved r.preTransition t h)

theorem forward_return {lib : APIHost api neg inputs Metadata} {metadata : Wrapper.MetadataAPI api neg inputs Metadata}
    {service : Runtime.Service platform api neg inputs Metadata} {identity : Nat}
    {x w : Storage.Tensor} {bias : Option Storage.Tensor} {eps : Compiler.Shared.HostScalar}
    {s : State Store Metadata} (f : ForwardCall.SuccessfulCall lib metadata service identity x w bias eps s) :
    TensorReady f.final f.body.output.tensor := by
  have hy := forward_output f.body.dispatch (2 : Fin 3)
  have hsave := preserved ((lib.save f.body.context).transition f.body.dispatch.final) _ hy
  have hv := view_ready lib f.body.preparation.data.y f.body.output x.view.shape f.body.sSave f.body.viewed hsave
  exact preserved ((lib.leave f.sShape.owner).transition f.body.final) _ hv

theorem backward_return {lib : APIHost api neg inputs Metadata}
    {service : BackwardRuntime.Service platform api neg inputs Metadata} {identity : Nat}
    {c : SavedContext} {saved : BackwardPreparation.SavedLayout c} {dy : Storage.Tensor}
    {s : State Store Metadata} (b : BackwardCall.SuccessfulCall lib service identity c saved dy s)
    (out : Storage.Tensor) (ho : out∈GradientReturn.returned b.body.output) : TensorReady b.final out := by
  apply preserved ((lib.leave b.accessState.owner).transition b.body.final)
  apply return_ready b.body.returns _ out ho
  intro t ht
  simp only [BackwardLayout.outputs,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with (hx|hw)|hb
  · subst t
    exact backward_preserved b.body.affine _
      (backward_output b.body.dx ⟨0,by simp [BackwardRuntime.Kind.count]⟩)
  · subst t
    simpa [BackwardRuntime.Kind.output,BackwardLayout.affineOutput] using backward_output b.body.affine ⟨0,by simp [BackwardRuntime.Kind.count,BackwardLayout.affineCount]⟩
  · have he : b.body.preparation.data.db=some t := by simpa using hb
    simpa [BackwardRuntime.Kind.output,BackwardLayout.affineOutput,he] using
      backward_output b.body.affine ⟨1,by simp [BackwardRuntime.Kind.count,BackwardLayout.affineCount,he]⟩

#axiomsClean forward_return
#axiomsClean backward_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Readiness
