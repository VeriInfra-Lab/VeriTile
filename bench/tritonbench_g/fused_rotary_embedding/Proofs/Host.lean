import bench.tritonbench_g.fused_rotary_embedding.Proofs.Metadata
import bench.tritonbench_g.fused_rotary_embedding.Proofs.Bounds
import bench.tritonbench_g.fused_rotary_embedding.Proofs.Logical

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding.Host
structure Call where
  config : Config
  identities : Region → Nat
  device : Nat

/-- Canonical shapes/strides and metadata checks are implemented by the host.
Ordinary independent allocation provenance and device lifetime are runtime
facts, not numerical or output-valued assumptions. -/
structure Admitted (call : Call) : Prop where
  config : call.config.Valid
  ordinarySplit : call.config.shape.newLayout=false → call.config.shape.split=call.config.shape.width
  dataBounds : ∀ kind,call.config.shape.tokens*call.config.shape.heads kind*call.config.shape.width≤(2^63-1)/4
  cacheBounds : call.config.shape.blocks*call.config.shape.headsK*call.config.shape.blockSize*call.config.shape.width≤(2^63-1)/4
  cosBounds : call.config.shape.tokens*call.config.shape.cosWidth≤(2^63-1)/4
  sinBounds : call.config.shape.tokens*call.config.shape.sinWidth≤(2^63-1)/4
  tableBounds : call.config.shape.tokens*call.config.shape.tableWidth≤(2^63-1)/4
  lengthBounds : call.config.shape.tokens≤(2^63-1)/4

structure Launch where
  heads : Nat
  tokens : Nat
  groups : Nat
  blockHalf : Nat
  blockDim : Nat
  warps : Nat
  deriving DecidableEq,Repr
def launch (s : Shape) : Launch :=
  ⟨s.headsQ,s.tokens,s.groups,s.halfBlock,s.fullBlock,if 512≤s.width then 16 else if 256≤s.width then 8 else 4⟩
def empty (s : Shape) : Bool := decide (s.tokens=0)
structure Returned (α : Type) where
  value : Option Unit
  identities : Region → Nat
  deviceRestored : Nat
  memory : Memory α
noncomputable def mathematical (call : Call) (callerDevice : Nat) (m : Memory ℝ) : Returned ℝ :=
  ⟨none,call.identities,callerDevice,if empty call.config.shape then m else actualSource call.config (fun _ => realOps) m⟩
def selected (call : Call) (callerDevice : Nat) (p : Native.Instructions) (m : Physical.Bytes) : Returned Physical.Byte :=
  ⟨none,call.identities,callerDevice,if empty call.config.shape then m else Physical.dynamic call.config.shape p m⟩

theorem empty_schedule (s : Shape) (h : empty s=true) : schedule s=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w hw
  have a := (schedule_active s w hw).1
  have t : s.tokens=0 := by simpa [empty] using h
  omega
theorem mathematical_memory (call : Call) (valid : call.config.Valid) (callerDevice : Nat) (m : Memory ℝ) :
    (mathematical call callerDevice m).memory=actualSource call.config (fun _ => realOps) m := by
  unfold mathematical
  by_cases h : empty call.config.shape=true
  · simp only [h,ite_true]
    rw [actualSource_eq_run call.config valid,run,empty_schedule call.config.shape h]
    rfl
  · simp [h]
theorem selected_memory (call : Call) (callerDevice : Nat) (p : Native.Instructions) (m : Physical.Bytes) :
    (selected call callerDevice p m).memory=Physical.dynamic call.config.shape p m := by
  unfold selected
  by_cases h : empty call.config.shape=true
  · simp only [h,ite_true]
    rw [Physical.dynamic,empty_schedule call.config.shape h]
    rfl
  · simp [h]
theorem returned_objects (call : Call) (callerDevice : Nat) (p : Native.Instructions) (m : Physical.Bytes) :
    (selected call callerDevice p m).value=none ∧
    (selected call callerDevice p m).identities=call.identities ∧
    (selected call callerDevice p m).deviceRestored=callerDevice := ⟨rfl,rfl,rfl⟩
theorem no_launch_no_writes (call : Call) (callerDevice : Nat) (p : Native.Instructions)
    (m : Physical.Bytes) (h : empty call.config.shape=true) : (selected call callerDevice p m).memory=m := by
  simp [selected,h]
theorem configuration_binding (call : Call) (m : Physical.Bytes)
    (binding : Physical.metadata m=call.config.metadata) : Physical.configuration call.config.shape m=call.config := by
  cases call with
  | mk c ids dev => cases c;simp_all [Physical.configuration]

theorem input_int64 (call : Call) (admitted : Admitted call) (w : Work)
    (hw : call.config.shape.Active w) (o : Bool) (i : Fin 4) :
    4*call.config.shape.input w o+i.val<2^63 :=
  active_byte_int64 _ _ (input_bound _ admitted.config.shape w hw o) (admitted.dataBounds w.kind) i
theorem output_int64 (call : Call) (admitted : Admitted call) (w : Work)
    (hw : call.config.shape.Active w) (o : Bool) (i : Fin 4) : 4*call.config.output w o+i.val<2^63 := by
  apply active_byte_int64 _ _ (output_bound _ admitted.config w hw o) _ i
  split
  · simpa [Shape.heads] using admitted.dataBounds .q
  · exact admitted.cacheBounds
theorem table_int64 (call : Call) (admitted : Admitted call) (t : Nat)
    (ht : t<call.config.shape.tokens) (i : Fin 4) :
    4*(t*call.config.shape.tableWidth+call.config.past t/call.config.shape.blockSize)+i.val<2^63 :=
  active_byte_int64 _ _ (table_bound _ admitted.config t ht) admitted.tableBounds i
theorem coefficient_int64 (call : Call) (admitted : Admitted call) (w : Work)
    (hw : call.config.shape.Active w) (sine : Bool) (i : Fin 4) :
    4*call.config.shape.coeff sine w.token w.pair+i.val<2^63 := by
  apply active_byte_int64 _ _ (coefficient_bound _ admitted.config.shape w hw sine) _ i
  cases sine
  · exact admitted.cosBounds
  · exact admitted.sinBounds

#axiomsClean returned_objects
#axiomsClean no_launch_no_writes
#axiomsClean input_int64
#axiomsClean output_int64
#axiomsClean table_int64
end VeriTile.Bench.Repairs.FusedRotaryEmbedding.Host
