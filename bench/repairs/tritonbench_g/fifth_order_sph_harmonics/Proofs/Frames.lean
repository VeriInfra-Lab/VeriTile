import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Host

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Frames
open VeriTile.Triton FP.Structural Packing Storage Host
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 64000

theorem gradient_span (x u : TensorView) (g : TensorObject) (fresh : FreshGradient x u g) (hn : 0<x.rows) :
    g.metadata.storageBase≤g.metadata.pointer ∧
    g.metadata.pointer+12*x.rows≤g.metadata.storageBase+g.metadata.capacity := by
  obtain ⟨_,_,_,_,_,bounds⟩:=fresh.bounds
  have rows:=fresh.rows
  have columns:=fresh.columns
  have safe:=bounds ⟨by omega,by omega⟩
  simp only [TensorView.rowStride,TensorView.colStride,fresh.strides] at safe
  norm_num at safe
  rw [rows,columns] at safe
  constructor
  · exact safe.2.1
  · have upper:=safe.2.2.1
    change g.metadata.pointer+4*((x.rows-1)*3+2+1)≤g.metadata.storageBase+g.metadata.capacity at upper
    omega

theorem outside_gradient (x u : TensorView) (g : TensorObject) (fresh : FreshGradient x u g)
    (t : TensorView) (valid : TensorBounds t)
    (apart : g.metadata.storageBase+g.metadata.capacity≤t.storageBase ∨
      t.storageBase+t.capacity≤g.metadata.storageBase)
    (r c : Nat) (hr : r<t.rows) (hc : c<t.columns) (nonempty : 0<x.rows) (k : Fin 4) :
    Access.address t r c+k.val<g.metadata.pointer ∨
      g.metadata.pointer+12*x.rows≤Access.address t r c+k.val := by
  have input:=Access.storage_access t valid r c hr hc
  have output:=gradient_span x u g fresh nonempty
  have hk:=k.isLt
  rcases apart with apart|apart <;> omega

theorem zero_input (x u : TensorView) (g : TensorObject) (fresh : FreshGradient x u g)
    (t : TensorView) (valid : TensorBounds t)
    (apart : g.metadata.storageBase+g.metadata.capacity≤t.storageBase ∨
      t.storageBase+t.capacity≤g.metadata.storageBase)
    (r c : Nat) (hr : r<t.rows) (hc : c<t.columns) (nonempty : 0<x.rows) (heap : Heap) :
    Storage.read (zeroGradient g.metadata.pointer x.rows heap) (Access.address t r c)=
      Storage.read heap (Access.address t r c) := by
  funext k
  have outside:=outside_gradient x u g fresh t valid apart r c hr hc nonempty k
  simp only [Storage.read,zeroGradient]
  rw [if_neg (by omega)]

theorem forward_inputs (M : Algebra Word) (q : Request) (admit : Admitted q)
    (allocated : TensorObject) (allocation : Allocation q allocated) (heap : Heap)
    (r : Nat) (hr : r<q.coordinates.metadata.rows) (c : Fin 3) (k : Fin 4) :
    (Host.forward M q admit allocated heap).2.2 (q.coordinates.metadata.pointer+4*(r*3+c.val)+k.val)=
      heap (q.coordinates.metadata.pointer+4*(r*3+c.val)+k.val) := by
  have h:=Physical.complete M .forward (parameters q admit allocated) (forwardBuffers q allocated) heap
    (parameters_fit q admit allocated allocation)
  apply h.2.2
  intro s hs j byte same
  have dest:=destination q admit allocated allocation
  have apart:=Access.input_output_disjoint q.coordinates.metadata (q.output allocated).metadata q.column admit.1 dest
    r s c.val (q.column.toNat+j.val) k.val byte.val hr hs c.isLt
    (by have fit:=dest.2.2.1;have hj : j.val<11 :=j.isLt;omega) k.isLt byte.isLt
  apply apart
  rw [Access.coordinate_address _ admit.1,Access.destination_address _ _ _ dest]
  simpa [forwardBuffers,parameters,Grid.offset,Grid.stride,Grid.first,Nat.add_assoc] using same

theorem empty_launch (M : Algebra Word) (d : Grid.Direction) (p : Parameters) (b : Physical.Buffers)
    (heap : Heap) (empty : p.rows=0) : (Physical.launch M d p b heap).bytes=heap := by
  have grid:Grid.blocks p=0 := by
    simp only [Grid.blocks,empty,Nat.zero_add]
    exact Nat.div_eq_of_lt (by have hb:=p.blockPositive;omega)
  simp [Physical.launch,Physical.execute,grid,SourceBridge.run]

theorem backward_outside (M : Algebra Word) (p : Parameters) (ctx : Context)
    (upstream gradient : TensorObject) (heap : Heap) (fit:p.columnOffset+11≤p.outputWidth)
    (address : Nat) (outside:address<gradient.metadata.pointer ∨ gradient.metadata.pointer+12*p.rows≤address) :
    (Host.backward M p ctx upstream gradient heap).2 address=heap address := by
  have h:=Physical.complete M .backward (backwardParameters p upstream.metadata)
    (backwardBuffers ctx upstream gradient) (zeroGradient gradient.metadata.pointer p.rows heap) fit
  change (Physical.launch M .backward (backwardParameters p upstream.metadata)
    (backwardBuffers ctx upstream gradient) (zeroGradient gradient.metadata.pointer p.rows heap)).bytes address=heap address
  rw [h.2.2 address]
  · simp only [zeroGradient]
    rw [if_neg (by omega)]
  · intro r hr c k same
    have hc:c.val<3:=c.isLt
    have hk:=k.isLt
    change r<p.rows at hr
    change address=gradient.metadata.pointer+4*(r*3+c.val)+k.val at same
    omega

theorem backward_input (M : Algebra Word) (q : Request) (admit : Admitted q)
    (allocated : TensorObject) (allocation : Allocation q allocated) (upstream gradient : TensorObject)
    (fresh : FreshGradient q.coordinates.metadata upstream.metadata gradient)
    (t : TensorView) (valid : TensorBounds t)
    (apart : gradient.metadata.storageBase+gradient.metadata.capacity≤t.storageBase ∨
      t.storageBase+t.capacity≤gradient.metadata.storageBase)
    (heap : Heap) (r c : Nat) (hr:r<t.rows) (hc:c<t.columns)
    (nonempty:0<q.coordinates.metadata.rows) (k:Fin 4) :
    (Host.backward M (parameters q admit allocated) (save q allocated) upstream gradient heap).2
      (Access.address t r c+k.val)=heap (Access.address t r c+k.val) :=
  backward_outside M _ _ _ _ _ (parameters_fit q admit allocated allocation) _
    (outside_gradient _ _ _ fresh t valid apart r c hr hc nonempty k)

#axiomsClean gradient_span
#axiomsClean zero_input
#axiomsClean forward_inputs
#axiomsClean empty_launch
#axiomsClean backward_outside
#axiomsClean backward_input
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Frames
