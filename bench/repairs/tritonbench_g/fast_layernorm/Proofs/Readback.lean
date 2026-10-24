import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Runtime

namespace VeriTile.Bench.Repairs.FastLayernorm.Runtime
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Execution Compiler Layout
noncomputable section
open scoped Classical

theorem forward_buffer (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (m : Storage.World Store) (k : Fin 3) :
    ((forwardTrace ops plan api p layout eps m).regions k).buffer=outputs p k := by
  cases hb : p.bias <;> fin_cases k <;> simp [forwardTrace,hb,forwardRegions,forwardRegionsNoBias,outputs]

def biasValues (ops : Ops α) (api : Memory.API α Store Byte) (p : Data) (m : Storage.World Store) : Nat → α :=
  match p.bias with | none => fun _ => ops.zero | some b => vector b m api

theorem forward_y_value (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (m : Storage.World Store)
    (i : Fin ((forwardTrace ops plan api p layout eps m).regions (2 : Fin 3)).buffer.count) :
    ((forwardTrace ops plan api p layout eps m).regions (2 : Fin 3)).value i =
      forwardReference ops plan (rows p.x N m api (i.val/N)) (vector p.w m api)
        (biasValues ops api p m) eps (i.val%N) := by
  rcases p with ⟨x,w,y,r,mu,bias⟩
  cases bias <;> rfl

theorem forward_mean_value (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (m : Storage.World Store)
    (i : Fin ((forwardTrace ops plan api p layout eps m).regions (1 : Fin 3)).buffer.count) :
    ((forwardTrace ops plan api p layout eps m).regions (1 : Fin 3)).value i =
      (statistics ops plan (rows p.x N m api i.val) eps 0).mean := by
  rcases p with ⟨x,w,y,r,mu,bias⟩
  cases bias <;> rfl

theorem forward_inverse_value (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (m : Storage.World Store)
    (i : Fin ((forwardTrace ops plan api p layout eps m).regions (0 : Fin 3)).buffer.count) :
    ((forwardTrace ops plan api p layout eps m).regions (0 : Fin 3)).value i =
      (statistics ops plan (rows p.x N m api i.val) eps 0).inverse := by
  rcases p with ⟨x,w,y,r,mu,bias⟩
  cases bias <;> rfl

theorem forward_y_readback (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (schedule : Schedule 3)
    (covered : ∀ k i,i<(outputs p k).count → (k,i)∈schedule.order)
    (m : Storage.World Store) (i : Fin p.y.count) :
    api.read (forwardMemory ops plan api p layout eps schedule m p.x.owner) (p.y.address i.val)=
      forwardReference ops plan (rows p.x N m api (i.val/N)) (vector p.w m api)
        (biasValues ops api p m) eps (i.val%N) := by
  let t := forwardTrace ops plan api p layout eps m
  have hc : ∀ k j,j<(t.regions k).buffer.count → (k,j)∈schedule.order := by
    intro k j hj
    exact covered k j (by simpa only [t,forward_buffer] using hj)
  have hi : i.val<(t.regions (2 : Fin 3)).buffer.count := by
    simpa only [t,forward_buffer,outputs] using i.isLt
  have h := scheduled_readback api t schedule hc m (2 : Fin 3) ⟨i.val,hi⟩
  have hv := h.trans (forward_y_value ops plan api p layout eps m ⟨i.val,hi⟩)
  change api.read (forwardMemory ops plan api p layout eps schedule m p.x.owner)
    ((t.regions (2 : Fin 3)).buffer.address i.val) = _ at hv
  simp only at hv
  rw [show (t.regions (2 : Fin 3)).buffer=p.y from forward_buffer ops plan api p layout eps m 2] at hv
  exact hv

theorem forward_mean_readback (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (schedule : Schedule 3)
    (covered : ∀ k i,i<(outputs p k).count → (k,i)∈schedule.order)
    (m : Storage.World Store) (i : Fin p.mu.count) :
    api.read (forwardMemory ops plan api p layout eps schedule m p.x.owner) (p.mu.address i.val)=
      (statistics ops plan (rows p.x N m api i.val) eps 0).mean := by
  let t := forwardTrace ops plan api p layout eps m
  have hc : ∀ k j,j<(t.regions k).buffer.count → (k,j)∈schedule.order := by
    intro k j hj
    exact covered k j (by simpa only [t,forward_buffer] using hj)
  have hi : i.val<(t.regions (1 : Fin 3)).buffer.count := by
    simpa only [t,forward_buffer,outputs] using i.isLt
  have h := scheduled_readback api t schedule hc m (1 : Fin 3) ⟨i.val,hi⟩
  have hv := h.trans (forward_mean_value ops plan api p layout eps m ⟨i.val,hi⟩)
  change api.read (forwardMemory ops plan api p layout eps schedule m p.x.owner)
    ((t.regions (1 : Fin 3)).buffer.address i.val) = _ at hv
  simp only at hv
  rw [show (t.regions (1 : Fin 3)).buffer=p.mu from forward_buffer ops plan api p layout eps m 1] at hv
  exact hv

theorem forward_inverse_readback (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : Data) (layout : Forward p rowCount N inputs) (eps : α) (schedule : Schedule 3)
    (covered : ∀ k i,i<(outputs p k).count → (k,i)∈schedule.order)
    (m : Storage.World Store) (i : Fin p.r.count) :
    api.read (forwardMemory ops plan api p layout eps schedule m p.x.owner) (p.r.address i.val)=
      (statistics ops plan (rows p.x N m api i.val) eps 0).inverse := by
  let t := forwardTrace ops plan api p layout eps m
  have hc : ∀ k j,j<(t.regions k).buffer.count → (k,j)∈schedule.order := by
    intro k j hj
    exact covered k j (by simpa only [t,forward_buffer] using hj)
  have hi : i.val<(t.regions (0 : Fin 3)).buffer.count := by
    simpa only [t,forward_buffer,outputs] using i.isLt
  have h := scheduled_readback api t schedule hc m (0 : Fin 3) ⟨i.val,hi⟩
  have hv := h.trans (forward_inverse_value ops plan api p layout eps m ⟨i.val,hi⟩)
  change api.read (forwardMemory ops plan api p layout eps schedule m p.x.owner)
    ((t.regions (0 : Fin 3)).buffer.address i.val) = _ at hv
  simp only at hv
  rw [show (t.regions (0 : Fin 3)).buffer=p.r from forward_buffer ops plan api p layout eps m 0] at hv
  exact hv

variable {platform : Compiler.Platform} {api : Memory.API UInt32 Store UInt8} {neg : UInt32 → UInt32}
  {inputs : List Storage.Tensor} {Metadata : Type} {rowCount N B : Nat}
  {service : Service platform api neg inputs Metadata} {p : Data}
  {layout : Forward p rowCount N inputs} {settings : Compiler.settings N=.ok (B,Compiler.warps B)}
  {eps : Compiler.Shared.HostScalar} {initial : State Store Metadata}

theorem route_coverage (r : Route service p layout settings eps initial) :
    ∀ k i,i<(outputs p k).count → (k,i)∈r.schedule.order :=
  (service.admitted p layout r.code r.s1 r.schedule r.admitted).2.2.2.2.2 r.completed

theorem route_y_readback (r : Route service p layout settings eps initial) (i : Fin p.y.count) :
    api.read (r.final.memory p.x.owner) (p.y.address i.val)=
      forwardReference (platform.scalar r.code.object.primitives) r.code.plan
        (rows p.x N r.s2.memory api (i.val/N)) (vector p.w r.s2.memory api)
        (biasValues (platform.scalar r.code.object.primitives) api p r.s2.memory)
        (platform.epsilon r.code.object eps) (i.val%N) :=
  forward_y_readback _ _ _ _ _ _ _ (route_coverage r) _ i

theorem route_real_y_readback (r : Route service p layout settings eps initial)
    (ops : Ops ℝ) (m : Storage.World (Nat → ℝ)) (epsReal : ℝ) (i : Fin p.y.count) :
    r.realFinal ops m epsReal p.x.owner (p.y.address i.val)=
      forwardReference ops r.code.plan
        (rows p.x N (r.preTransition.shadow m) Memory.realAPI (i.val/N))
        (vector p.w (r.preTransition.shadow m) Memory.realAPI)
        (biasValues ops Memory.realAPI p (r.preTransition.shadow m)) epsReal (i.val%N) :=
  forward_y_readback _ _ Memory.realAPI _ _ _ _ (route_coverage r) _ i

#axiomsClean forward_y_readback
#axiomsClean forward_mean_readback
#axiomsClean forward_inverse_readback
#axiomsClean route_y_readback
#axiomsClean route_real_y_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Runtime
