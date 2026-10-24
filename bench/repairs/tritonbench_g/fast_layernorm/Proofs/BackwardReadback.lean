import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardRuntime
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Real

namespace VeriTile.Bench.Repairs.FastLayernorm.BackwardRuntime
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Execution BackwardPreparation BackwardLayout
noncomputable section
open scoped Classical

def sourceValue (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block)
    (api : Memory.API α Store Byte) (p : Data) (kind : Kind) (m : Storage.World Store) (k col : Nat) : α :=
  match kind with
  | .dx => backwardReference ops plan (Execution.saved api c.r c.mu m (col/c.columns))
      (rows c.x c.columns m api (col/c.columns)) (rows p.dy c.columns m api (col/c.columns))
      (vector c.w m api) (col%c.columns)
  | .affine =>
    let pair := affineReference ops plan c.rows (rows c.x c.columns m api)
      (rows p.dy c.columns m api) (Execution.saved api c.r c.mu m) col
    if k=0 then pair.1 else pair.2

theorem trace_owner (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block)
    (api : Memory.API α Store Byte) (p : Data) (layout : Layout c p inputs)
    (kind : Kind) (m : Storage.World Store) : (trace c ops plan api p layout kind m).owner=c.x.owner := by
  cases kind <;> rfl

theorem trace_value (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block)
    (api : Memory.API α Store Byte) (p : Data) (layout : Layout c p inputs) (kind : Kind) (m : Storage.World Store)
    (k : Fin (kind.count p)) (i : Fin ((trace c ops plan api p layout kind m).regions k).buffer.count) :
    ((trace c ops plan api p layout kind m).regions k).value i=sourceValue c ops plan api p kind m k.val i.val := by
  cases kind <;> rfl

theorem kernel_readback (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block)
    (api : Memory.API α Store Byte) (p : Data) (layout : Layout c p inputs) (kind : Kind)
    (schedule : Schedule (kind.count p))
    (covered : ∀ k i,i<(kind.output p k).count → (k,i)∈schedule.order)
    (m : Storage.World Store) (k : Fin (kind.count p)) (i : Fin (kind.output p k).count) :
    api.read (kernelMemory c ops plan api p layout kind schedule m c.x.owner) ((kind.output p k).address i.val)=
      sourceValue c ops plan api p kind m k.val i.val := by
  let t := trace c ops plan api p layout kind m
  have hc : ∀ k i,i<(t.regions k).buffer.count → (k,i)∈schedule.order := by
    intro k i hi
    exact covered k i (by simpa only [t,trace_buffer] using hi)
  have hi : i.val<(t.regions k).buffer.count := by simpa only [t,trace_buffer] using i.isLt
  have h := scheduled_readback api t schedule hc m k ⟨i.val,hi⟩
  have hv := h.trans (trace_value c ops plan api p layout kind m k ⟨i.val,hi⟩)
  simp only at hv
  rw [show (t.regions k).buffer=kind.output p k from trace_buffer c ops plan api p layout kind m k] at hv
  simpa only [t,kernelMemory,trace_owner] using hv

theorem preserved_input (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block)
    (api : Memory.API α Store Byte) (p : Data) (layout : Layout c p inputs) (kind : Kind)
    (schedule : Schedule (kind.count p)) (m : Storage.World Store)
    (input : Dense) (member : input.tensor∈inputs) (owner : input.owner=c.x.owner) (i : Fin input.count) :
    api.read (kernelMemory c ops plan api p layout kind schedule m input.owner) (input.address i.val)=
      api.read (m input.owner) (input.address i.val) := by
  apply trace_prefix_input api (trace c ops plan api p layout kind m) m input
  · simpa only [trace_owner] using owner
  · intro k
    rw [trace_buffer]
    exact layout.callerFresh _ (output_mem kind p k) _ member
  · exact scheduled_subset _ schedule

theorem affine_preserves_dx (c : SavedContext) (ops : Ops α) (plan : Plan c.columns c.block)
    (api : Memory.API α Store Byte) (p : Data) (layout : Layout c p inputs)
    (schedule : Schedule (Kind.affine.count p)) (m : Storage.World Store) (i : Fin p.dx.count) :
    api.read (kernelMemory c ops plan api p layout .affine schedule m p.dx.owner) (p.dx.address i.val)=
      api.read (m p.dx.owner) (p.dx.address i.val) := by
  apply trace_prefix_input api (trace c ops plan api p layout .affine m) m p.dx
  · exact layout.owners p.dx (by simp [outputs])
  · intro k
    rw [trace_buffer]
    have hpair : (p.dx::(p.dw::p.db.toList)).Pairwise (fun a b => Storage.Disjoint a.tensor.backing b.tensor.backing) := layout.pairwise
    have hm : affineOutput p k∈p.dw::p.db.toList := by
      rcases p with ⟨dy,dx,dw,db⟩
      cases db <;> fin_cases k <;> simp [affineOutput]
    exact disjoint_symm _ _ ((List.pairwise_cons.mp hpair).1 _ hm)
  · exact scheduled_subset _ schedule

variable {platform : Compiler.Platform} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {Metadata : Type} {service : Service platform api neg inputs Metadata}
  {c : SavedContext} {p : Data} {layout : Layout c p inputs} {kind : Kind} {initial : State Store Metadata}

theorem route_readback (r : Route service c p layout kind initial)
    (k : Fin (kind.count p)) (i : Fin (kind.output p k).count) :
    api.read (r.final.memory c.x.owner) ((kind.output p k).address i.val)=
      sourceValue c (platform.scalar r.code.object.primitives) r.code.plan api p kind r.s2.memory k.val i.val :=
  kernel_readback c _ _ _ _ _ _ _ (route_coverage r) _ k i

theorem route_real_readback (r : Route service c p layout kind initial)
    (m : Storage.World (Nat → ℝ)) (k : Fin (kind.count p)) (i : Fin (kind.output p k).count) :
    r.realFinal RealProof.ops m c.x.owner ((kind.output p k).address i.val)=
      sourceValue c RealProof.ops r.code.plan Memory.realAPI p kind (r.preTransition.shadow m) k.val i.val :=
  kernel_readback c RealProof.ops r.code.plan Memory.realAPI p layout kind r.schedule (route_coverage r) (r.preTransition.shadow m) k i

#axiomsClean kernel_readback
#axiomsClean preserved_input
#axiomsClean affine_preserves_dx
#axiomsClean route_readback
#axiomsClean route_real_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.BackwardRuntime
