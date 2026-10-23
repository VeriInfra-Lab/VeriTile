import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Real
import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Native

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PhysicalKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- Only raw pointer windows, scalar strides and current signed CSR reads. -/
structure RawABI (p : Parameters) (batches : Nat) (access : Access α Store Byte)
    (buffers : Physical.Buffers) (initial : Store) : Prop where
  heads : 0<p.heads
  stride : 16≤p.om
  parameters : p.ob=p.heads*p.length*(16*p.blocks) ∧ p.oh=p.length*(16*p.blocks) ∧ p.om=16*p.blocks
  window : ∀ w,w.active (shape p batches) →
    buffers.output.base≤Physical.outputAddress buffers (shape p batches) w ∧
    Physical.outputAddress buffers (shape p batches) w+4≤buffers.output.base+buffers.output.storage.bytes
  separate : ∀ r,buffers.output.base+buffers.output.storage.bytes≤(buffers.input r).base ∨
    (buffers.input r).base+(buffers.input r).storage.bytes≤buffers.output.base
  first : ∀ zero id,id.Valid p batches → 0≤KernelOnly.first p (state access zero buffers initial id)
  columns : ∀ zero id,id.Valid p batches → ∀ visit,
    visit<visitCount p (state access zero buffers initial id) →
      0≤csrColumn p (state access zero buffers initial id)
        (KernelOnly.first p (state access zero buffers initial id)+visit)

theorem raw_events_bounded (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial)
    (id : Program) (valid : id.Valid p batches) :
    Bounded buffers.output (events access a native p buffers initial id) := by
  intro e member
  change e∈(lanes p (state access a.zero buffers initial id)).map _ at member
  obtain ⟨i,hi,rfl⟩ := List.mem_map.mp member
  have live : id.row*16+i.1.val<p.length := of_decide_eq_true (List.mem_filter.mp hi).2
  have active := coordinate_active p batches id valid (show 0<p.heads from abi.heads) i live
  have bound := abi.window
    (coordinate p batches id i) active
  simpa only [output_offset access a.zero buffers initial p batches id i abi.parameters] using bound


theorem raw_grid_static (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial)
    (ids : List Program) (valid : ∀ id∈ids,id.Valid p batches)
    (current : Store) (stable : Stable access a.zero buffers initial current) :
    grid access a native p buffers ids current=some
      (LayernormFwdTriton.Memory.run access.toAPI (gridEvents access a native p buffers initial ids) current) := by
  induction ids generalizing current with
  | nil => rfl
  | cons id ids ih =>
    have hv := valid id (List.mem_cons_self ..)
    have bound := raw_events_bounded access a native p batches buffers initial abi id hv
    have hs := run_stable access a.zero buffers _ bound abi.separate initial current stable
    simp only [grid,program,trace_stable access a native p buffers initial current stable id
      abi.stride,Option.map_some,Option.bind_some]
    rw [ih (fun j hj => valid j (List.mem_cons_of_mem _ hj)) _ hs]
    simp only [gridEvents,List.flatMap_cons,LayernormFwdTriton.Memory.run,List.foldl_append]

theorem raw_grid_success (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial) :
    grid access a native p buffers (programs p batches) initial=some
      (LayernormFwdTriton.Memory.run access.toAPI
        (gridEvents access a native p buffers initial (programs p batches)) initial) :=
  raw_grid_static access a native p batches buffers initial abi _
    (fun id member => (programs_mem p batches id).mp member) initial (Stable.refl ..)

theorem raw_gridEvents_target (access : Access α Store Byte) (a : Arithmetic α) (native : Bool)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial) (f : Work → α)
    (values : ∀ id, id.Valid p batches → ∀ i : TileIndex [16,16],id.row*16+i.1.val<p.length →
      output a native p (state access a.zero buffers initial id) i (owner i.2.1.val)=
        f (coordinate p batches id i)) :
    gridEvents access a native p buffers initial (programs p batches)=targetTrace p batches buffers f := by
  unfold gridEvents targetTrace
  apply List.flatMap_congr
  intro id member
  change List.map _ (activeLanes p id)=List.map _ (activeLanes p id)
  apply List.map_congr_left
  intro i hi
  have live := (activeLanes_mem p id i).mp hi
  have value := values id ((programs_mem p batches id).mp member) i live
  simp only [output_offset access a.zero buffers initial p batches id i abi.parameters,value]
  rfl


theorem raw_real_events (access : Access R Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial)
    (finite : FiniteInputs access buffers initial) :
    gridEvents access mathArithmetic Bool.false p buffers initial (programs p batches)=
      targetTrace p batches buffers (fun w => some (realTarget access p batches buffers initial w)) := by
  apply raw_gridEvents_target access mathArithmetic Bool.false p batches buffers initial abi
  intro id valid i live
  have views := state_views access (some 0) buffers initial id
  rw [realMemory_lift access buffers initial finite] at views
  exact output_real p batches (state access (some 0) buffers initial id)
    (realMemory access buffers initial) _ _ views
    (abi.first (some 0) id valid)
    (abi.columns (some 0) id valid) i live

/-- Successful full shared-quote replay gives the independent causal CSR
occurrence-multiset attention quotient over every channel and output cell. -/
theorem raw_whole_real (access : Access R Store Byte) (p : Parameters) (batches : Nat)
    (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial)
    (finite : FiniteInputs access buffers initial) :
    ∃ final, grid access mathArithmetic Bool.false p buffers (programs p batches) initial=some final ∧
      (∀ w,w.active (shape p batches) →
        access.read final (Physical.outputAddress buffers (shape p batches) w)=
          some (realTarget access p batches buffers initial w)) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  refine ⟨LayernormFwdTriton.Memory.run access.toAPI
    (targetTrace p batches buffers (fun w => some (realTarget access p batches buffers initial w))) initial,?_,?_,?_⟩
  · rw [raw_grid_success access mathArithmetic Bool.false p batches buffers initial abi,
      raw_real_events access p batches buffers initial abi finite]
    rfl
  · exact fun w live => target_readback access.toAPI p batches buffers _ abi.heads initial w live
  · exact fun address outside => target_frame access.toAPI p batches buffers _ abi.heads initial address outside

theorem raw_native_events (ops : NativePrimitives) (access : Access Word Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial) :
    gridEvents access ops.arithmetic Bool.true p buffers initial (programs p batches)=
      targetTrace p batches buffers (nativeTarget ops access p batches buffers initial) := by
  apply raw_gridEvents_target access ops.arithmetic Bool.true p batches buffers initial abi
  intro id valid i _
  exact output_view (BlockSparseAttn.Native.ops ops.instructions) ops.literal (fun _ => ops.scale) ops.zero ops.negativeInfinity
    p batches (state access 0 buffers initial id) (Physical.inputs access.toAPI 0 buffers initial)
    _ _ (state_views access 0 buffers initial id)
    (abi.first 0 id valid)
    (abi.columns 0 id valid) i

/-- Whole selected-native kernel execution, exact independent scalar target,
and every caller byte outside the active output windows. No successful-run
or whole-output equality is an input assumption. -/
theorem raw_whole_native (ops : NativePrimitives) (access : Access Word Store Byte)
    (p : Parameters) (batches : Nat) (buffers : Physical.Buffers) (initial : Store)
    (abi : RawABI p batches access buffers initial) :
    ∃ final, grid access ops.arithmetic Bool.true p buffers (programs p batches) initial=some final ∧
      (∀ w,w.active (shape p batches) →
        access.read final (Physical.outputAddress buffers (shape p batches) w)=
          nativeTarget ops access p batches buffers initial w) ∧
      (∀ address,Outside p batches buffers address → access.byte final address=access.byte initial address) := by
  refine ⟨LayernormFwdTriton.Memory.run access.toAPI
    (targetTrace p batches buffers (nativeTarget ops access p batches buffers initial)) initial,?_,?_,?_⟩
  · rw [raw_grid_success access ops.arithmetic Bool.true p batches buffers initial abi,
      raw_native_events ops access p batches buffers initial abi]
  · exact fun w live => target_readback access.toAPI p batches buffers _ abi.heads initial w live
  · exact fun address outside => target_frame access.toAPI p batches buffers _ abi.heads initial address outside

#axiomsClean raw_whole_real
#axiomsClean raw_whole_native
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.PhysicalKernel
