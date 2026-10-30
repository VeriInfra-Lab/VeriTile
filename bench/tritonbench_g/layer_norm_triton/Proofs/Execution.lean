import bench.tritonbench_g.layer_norm_triton.Proofs.Model
import bench.tritonbench_g.layer_norm_triton.Proofs.Geometry

/-! Actual dense FP32 writes. The generic storage library supplies the byte
semantics and frame proofs. First-stage scratch is read from the resulting
memory by the second launch; no gradient-output equation is an API premise. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Execution
open VeriTile.Bench.Repairs.LayernormFwdTriton
open VeriTile.Bench.Repairs.FastLayernorm hiding Ops Plan
namespace E
abbrev Region := VeriTile.Bench.Repairs.FastLayernorm.Execution.Region
abbrev Trace := VeriTile.Bench.Repairs.FastLayernorm.Execution.Trace
abbrev Schedule := VeriTile.Bench.Repairs.FastLayernorm.Execution.Schedule
abbrev rows := @VeriTile.Bench.Repairs.FastLayernorm.Execution.rows
abbrev vector := @VeriTile.Bench.Repairs.FastLayernorm.Execution.vector
abbrev saved := @VeriTile.Bench.Repairs.FastLayernorm.Execution.saved
end E
open VeriTile.Bench.Repairs.FastLayernorm.Storage
noncomputable section
open scoped Classical

structure Buffers (K : Nat) (inputs : List Storage.Tensor) where
  owner : Nat
  output : Fin K → Dense
  owned : ∀ k,(output k).owner=owner
  separated : ∀ i j,i≠j → Storage.Disjoint (output i).tensor.backing (output j).tensor.backing
  callerFresh : ∀ k old,old∈inputs → Storage.Disjoint (output k).tensor.backing old.backing

def Buffers.trace (p : Buffers K inputs) (values : ∀ k,Fin (p.output k).count → α) : E.Trace α K where
  owner := p.owner
  regions k := ⟨p.output k,values k⟩
  owned := p.owned
  separated := p.separated

def Buffers.run (p : Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (schedule : E.Schedule K)
    (m : Storage.World Store) : Storage.World Store :=
  Storage.runWorld api p.owner
    (VeriTile.Bench.Repairs.FastLayernorm.Execution.scheduledWrites (p.trace values) schedule) m

theorem run_frame (p : Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (schedule : E.Schedule K)
    (m : Storage.World Store) (key : Key) (hk : Storage.caller inputs key) :
    Storage.observe api (p.run api values schedule m) key=Storage.observe api m key := by
  apply VeriTile.Bench.Repairs.FastLayernorm.Execution.trace_prefix_frame api (p.trace values) m inputs
  · exact p.callerFresh
  · exact VeriTile.Bench.Repairs.FastLayernorm.Execution.scheduled_subset _ schedule
  · exact hk

theorem run_readback (p : Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (schedule : E.Schedule K)
    (coverage : ∀ k i,i<(p.output k).count → (k,i)∈schedule.order)
    (m : Storage.World Store) (k : Fin K) (i : Fin (p.output k).count) :
    api.read (p.run api values schedule m p.owner) ((p.output k).address i.val)=values k i :=
  VeriTile.Bench.Repairs.FastLayernorm.Execution.scheduled_readback api (p.trace values) schedule coverage m k i

theorem run_input (p : Buffers K inputs) (api : Memory.API α Store Byte)
    (values : ∀ k,Fin (p.output k).count → α) (schedule : E.Schedule K)
    (m : Storage.World Store) (input : Dense) (ho : input.owner=p.owner)
    (fresh : ∀ k,Storage.Disjoint (p.output k).tensor.backing input.tensor.backing) (i : Fin input.count) :
    api.read (p.run api values schedule m input.owner) (input.address i.val)=
      api.read (m input.owner) (input.address i.val) :=
  VeriTile.Bench.Repairs.FastLayernorm.Execution.trace_prefix_input api (p.trace values) m input ho fresh _
    (VeriTile.Bench.Repairs.FastLayernorm.Execution.scheduled_subset _ schedule) i

def forwardBuffers (p : VeriTile.Bench.Repairs.FastLayernorm.Preparation.Data)
    (layout : VeriTile.Bench.Repairs.FastLayernorm.Layout.Forward p M N inputs) : Buffers 3 inputs :=
  ⟨p.x.owner,VeriTile.Bench.Repairs.FastLayernorm.Layout.outputs p,
   layout.outputsOwned,layout.separated,layout.callerFresh⟩

def forwardValues (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : VeriTile.Bench.Repairs.FastLayernorm.Preparation.Data)
    (eps : α) (m : Storage.World Store) :
    ∀ k : Fin 3,Fin (VeriTile.Bench.Repairs.FastLayernorm.Layout.outputs p k).count → α
  | ⟨0,_⟩,i => (statistics ops plan (E.rows p.x N m api i.val) eps 0).inverse
  | ⟨1,_⟩,i => (statistics ops plan (E.rows p.x N m api i.val) eps 0).mean
  | ⟨2,_⟩,i => forwardValue ops plan (E.rows p.x N m api (i.val/N)) (E.vector p.w m api)
      (E.vector (p.bias.getD p.w) m api) eps (i.val%N)

/-- Backward first launch: DX followed by one final DW/DB write per group. -/
structure BackwardBuffers (M N G : Nat) (inputs : List Storage.Tensor) where
  buffers : Buffers 3 inputs
  dxCount : (buffers.output 0).count=M*N
  dwCount : (buffers.output 1).count=min G M*N
  dbCount : (buffers.output 2).count=min G M*N

def groupValues (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : BackwardBuffers M N G inputs) (x w dy r mu : Dense) (m : Storage.World Store) :
    ∀ k,Fin (p.buffers.output k).count → α
  | ⟨0,_⟩,i => backwardValue ops plan (E.saved api r mu m (i.val/N))
      (E.rows x N m api (i.val/N)) (E.rows dy N m api (i.val/N)) (E.vector w m api) (i.val%N)
  | ⟨1,_⟩,i => (groupValue ops plan M G (i.val/N) (E.rows x N m api) (E.rows dy N m api)
      (E.saved api r mu m) (i.val%N)).1
  | ⟨2,_⟩,i => (groupValue ops plan M G (i.val/N) (E.rows x N m api) (E.rows dy N m api)
      (E.saved api r mu m) (i.val%N)).2

structure FinalBuffers (N : Nat) (inputs : List Storage.Tensor) where
  buffers : Buffers 2 inputs
  dwCount : (buffers.output 0).count=N
  dbCount : (buffers.output 1).count=N

def finalValues (ops : Ops α) (plan : FinalPlan) (api : Memory.API α Store Byte)
    (p : FinalBuffers N inputs) (groups : Nat) (dw db : Dense) (m : Storage.World Store) :
    ∀ k,Fin (p.buffers.output k).count → α
  | ⟨0,_⟩,i => finalValue ops plan groups (fun group => E.rows dw N m api group i.val) i.val
  | ⟨1,_⟩,i => finalValue ops plan groups (fun group => E.rows db N m api group i.val) i.val

theorem group_readback (ops : Ops α) (plan : Plan N B) (api : Memory.API α Store Byte)
    (p : BackwardBuffers M N G inputs) (x w dy r mu : Dense) (m : Storage.World Store)
    (schedule : E.Schedule 3)
    (coverage : ∀ k i,i<(p.buffers.output k).count → (k,i)∈schedule.order)
    (group col : Nat) (hg : group<min G M) (hc : col<N) :
    let after := p.buffers.run api (groupValues ops plan api p x w dy r mu m) schedule m
    E.rows (p.buffers.output 1) N after api group col=
      (groupValue ops plan M G group (E.rows x N m api) (E.rows dy N m api) (E.saved api r mu m) col).1 ∧
    E.rows (p.buffers.output 2) N after api group col=
      (groupValue ops plan M G group (E.rows x N m api) (E.rows dy N m api) (E.saved api r mu m) col).2 := by
  have hd := (Geometry.partial_offset_safe (p.buffers.output 1) M N G group col p.dwCount hg hc).1
  have hb := (Geometry.partial_offset_safe (p.buffers.output 2) M N G group col p.dbCount hg hc).1
  have hdiv : (group*N+col)/N=group := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ (by omega),Nat.div_eq_of_lt hc,Nat.zero_add]
  have hmod : (group*N+col)%N=col := by simp [Nat.add_mod,Nat.mod_eq_of_lt hc]
  constructor
  · have hr := run_readback p.buffers api (groupValues ops plan api p x w dy r mu m) schedule coverage m 1 ⟨group*N+col,hd⟩
    simpa only [E.rows,VeriTile.Bench.Repairs.FastLayernorm.Execution.rows,p.buffers.owned,
      groupValues,hdiv,hmod] using hr
  · have hr := run_readback p.buffers api (groupValues ops plan api p x w dy r mu m) schedule coverage m 2 ⟨group*N+col,hb⟩
    simpa only [E.rows,VeriTile.Bench.Repairs.FastLayernorm.Execution.rows,p.buffers.owned,
      groupValues,hdiv,hmod] using hr

#axiomsClean run_frame
#axiomsClean run_readback
#axiomsClean run_input
#axiomsClean group_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Execution
