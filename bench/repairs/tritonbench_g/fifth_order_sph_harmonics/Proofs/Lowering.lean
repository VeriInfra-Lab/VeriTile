import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Host

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Lowering
open Host
set_option maxHeartbeats 3000000

theorem block_bounds (q : Request) (admit : Admitted q) : 0<q.block.toNat ∧ q.block.toNat≤1024 := by
  have hb:=admit.2.1
  unfold BlockValid at hb
  omega

theorem forward_masks (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (pid lane : Nat)
    (hp : pid<launchGrid q.coordinates.metadata.rows q.block.toNat) (hl : lane<q.block.toNat) :
    (∀ c : Fin 3,Native.mask q.coordinates.metadata.rows q.block.toNat 3 0 pid lane c.val=
      decide (pid*q.block.toNat+lane<q.coordinates.metadata.rows)) ∧
    (∀ c : Fin 11,Native.mask q.coordinates.metadata.rows q.block.toNat
      (q.output allocated).metadata.columns q.column.toNat pid lane c.val=
      decide (pid*q.block.toNat+lane<q.coordinates.metadata.rows)) := by
  have valid:=parameters_launch q admit allocated allocation
  have fit:=parameters_fit q admit allocated allocation
  dsimp only [parameters] at valid fit
  constructor
  · intro c
    exact Native.coordinate_mask _ _ _ _ _ _ _ valid (block_bounds q admit) c.isLt hp hl
  · intro c
    apply Native.mask_exact _ _ _ _ _ _ _ valid (block_bounds q admit) _ _ hp hl
    · have hc:=c.isLt
      change q.column.toNat+11≤(q.output allocated).metadata.columns at fit
      omega
    · exact Nat.le_of_lt_succ c.isLt

theorem forward_pointer (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (pid lane : Nat) (c : Fin 11)
    (hp : pid<launchGrid q.coordinates.metadata.rows q.block.toNat) (hl : lane<q.block.toNat)
    (active : pid*q.block.toNat+lane<q.coordinates.metadata.rows) :
    Native.pointer (q.output allocated).metadata.pointer q.block.toNat
      (q.output allocated).metadata.columns 1 q.column.toNat pid lane c.val=
      (q.output allocated).metadata.pointer+4*((pid*q.block.toNat+lane)*(q.output allocated).metadata.columns+q.column.toNat+c.val) := by
  let y:=(q.output allocated).metadata
  have dest:=destination q admit allocated allocation
  have safe:=Access.storage_access y dest.2.2.2.2.2.2.2.1
    (pid*q.block.toNat+lane) (q.column.toNat+c.val)
    (by have rows : y.rows=q.coordinates.metadata.rows :=dest.2.1;omega)
    (by have fit : q.column.toNat+11≤y.columns :=dest.2.2.1;have hc:=c.isLt;omega)
  rw [Access.destination_address q.coordinates.metadata y q.column dest] at safe
  have laneBounds:=Native.lane_bounds _ _ _ _ pid lane (parameters_launch q admit allocated allocation)
    (block_bounds q admit) hp hl
  have eq:=Native.pointer_exact y.pointer q.block.toNat y.columns 1 q.column.toNat pid lane c.val
    laneBounds.1 laneBounds.2.1 (by simp only [Nat.mul_one];omega)
  simpa [y,Nat.mul_one,Nat.add_assoc] using eq

theorem coordinate_pointer (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (pid lane : Nat) (c : Fin 3)
    (hp : pid<launchGrid q.coordinates.metadata.rows q.block.toNat) (hl : lane<q.block.toNat)
    (active : pid*q.block.toNat+lane<q.coordinates.metadata.rows) :
    Native.pointer q.coordinates.metadata.pointer q.block.toNat 3 1 0 pid lane c.val=
      q.coordinates.metadata.pointer+4*((pid*q.block.toNat+lane)*3+c.val) := by
  have valid:=admit.1
  have safe:=Access.storage_access q.coordinates.metadata valid.2.2.2 (pid*q.block.toNat+lane) c.val active
    (by have cols:=valid.2.1;have hc:=c.isLt;omega)
  rw [Access.coordinate_address q.coordinates.metadata valid] at safe
  have laneBounds:=Native.lane_bounds _ _ _ _ pid lane (parameters_launch q admit allocated allocation)
    (block_bounds q admit) hp hl
  simpa using Native.pointer_exact q.coordinates.metadata.pointer q.block.toNat 3 1 0 pid lane c.val
    laneBounds.1 laneBounds.2.1 (by simpa using (show q.coordinates.metadata.pointer+4*((pid*q.block.toNat+lane)*3+c.val)<2^64 by omega))

theorem upstream_pointer (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (upstream : TensorView)
    (upValid : UpstreamContract q.coordinates.metadata upstream (q.output allocated).metadata.columns)
    (pid lane : Nat) (c : Fin 11)
    (hp : pid<launchGrid q.coordinates.metadata.rows q.block.toNat) (hl : lane<q.block.toNat)
    (active : pid*q.block.toNat+lane<q.coordinates.metadata.rows) :
    Native.pointer upstream.pointer q.block.toNat upstream.rowStride.toNat upstream.colStride.toNat
      q.column.toNat pid lane c.val=
      Access.address upstream (pid*q.block.toNat+lane) (q.column.toNat+c.val) := by
  have shape:=upValid.2.1
  have rows:upstream.rows=q.coordinates.metadata.rows := by simp [TensorView.rows,shape]
  have cols:upstream.columns=(q.output allocated).metadata.columns := by simp [TensorView.columns,shape]
  have fit : q.column.toNat+11≤(q.output allocated).metadata.columns :=parameters_fit q admit allocated allocation
  have safe:=Access.storage_access upstream upValid.2.2.2 (pid*q.block.toNat+lane) (q.column.toNat+c.val)
    (by omega) (by have hc:=c.isLt;omega)
  have laneBounds:=Native.lane_bounds _ _ _ _ pid lane (parameters_launch q admit allocated allocation)
    (block_bounds q admit) hp hl
  exact Native.pointer_exact _ _ _ _ _ _ _ _ laneBounds.1 laneBounds.2.1 (by unfold Access.address at safe;omega)

theorem gradient_pointer (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (upstream : TensorView) (gradient : TensorObject)
    (fresh : FreshGradient q.coordinates.metadata upstream gradient) (pid lane : Nat) (c : Fin 3)
    (hp : pid<launchGrid q.coordinates.metadata.rows q.block.toNat) (hl : lane<q.block.toNat)
    (active : pid*q.block.toNat+lane<q.coordinates.metadata.rows) :
    Native.pointer gradient.metadata.pointer q.block.toNat 3 1 0 pid lane c.val=
      gradient.metadata.pointer+4*((pid*q.block.toNat+lane)*3+c.val) := by
  have rows:=fresh.rows
  have cols:=fresh.columns
  have safe:=Access.storage_access gradient.metadata fresh.bounds (pid*q.block.toNat+lane) c.val
    (by omega) (by have hc:=c.isLt;omega)
  have address : Access.address gradient.metadata (pid*q.block.toNat+lane) c.val=
      gradient.metadata.pointer+4*((pid*q.block.toNat+lane)*3+c.val) := by
    simp [Access.address,TensorView.rowStride,TensorView.colStride,fresh.strides]
  rw [address] at safe
  have laneBounds:=Native.lane_bounds _ _ _ _ pid lane (parameters_launch q admit allocated allocation)
    (block_bounds q admit) hp hl
  simpa using Native.pointer_exact gradient.metadata.pointer q.block.toNat 3 1 0 pid lane c.val
    laneBounds.1 laneBounds.2.1 (by simpa using (show gradient.metadata.pointer+4*((pid*q.block.toNat+lane)*3+c.val)<2^64 by omega))

#axiomsClean forward_masks
#axiomsClean forward_pointer
#axiomsClean coordinate_pointer
#axiomsClean upstream_pointer
#axiomsClean gradient_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Lowering
