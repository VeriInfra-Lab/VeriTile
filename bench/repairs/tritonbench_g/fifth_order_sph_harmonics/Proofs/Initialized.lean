import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Equivalence
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Frames

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Initialized
open VeriTile.Triton FP.Structural Packing Storage Host
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 64000

theorem coordinates (q : Request) (admit : Admitted q) (upstream gradient : TensorObject)
    (fresh : FreshGradient q.coordinates.metadata upstream.metadata gradient) (heap : Heap)
    (r : Nat) (hr:r<q.coordinates.metadata.rows) (c:Fin 3) :
    Storage.read (zeroGradient gradient.metadata.pointer q.coordinates.metadata.rows heap)
      (q.coordinates.metadata.pointer+4*(r*3+c.val))=
    Storage.read heap (q.coordinates.metadata.pointer+4*(r*3+c.val)) := by
  have h:=Frames.zero_input q.coordinates.metadata upstream.metadata gradient fresh q.coordinates.metadata
    admit.1.2.2.2 fresh.coordinatesApart r c.val hr (by have hc:=c.isLt;have cols:=admit.1.2.1;omega) (by omega) heap
  rw [Access.coordinate_address q.coordinates.metadata admit.1] at h
  exact h

theorem upstream (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (u gradient : TensorObject)
    (upValid : UpstreamContract q.coordinates.metadata u.metadata (q.output allocated).metadata.columns)
    (fresh : FreshGradient q.coordinates.metadata u.metadata gradient) (heap : Heap)
    (r : Nat) (hr:r<q.coordinates.metadata.rows) (c:Fin 11) :
    Storage.read (zeroGradient gradient.metadata.pointer q.coordinates.metadata.rows heap)
      (u.metadata.pointer+4*(r*u.metadata.rowStride.toNat+(q.column.toNat+c.val)*u.metadata.colStride.toNat))=
    Storage.read heap (u.metadata.pointer+4*(r*u.metadata.rowStride.toNat+(q.column.toNat+c.val)*u.metadata.colStride.toNat)) := by
  have shape:=upValid.2.1
  have rows:u.metadata.rows=q.coordinates.metadata.rows := by simp [TensorView.rows,shape]
  have cols:u.metadata.columns=(q.output allocated).metadata.columns := by simp [TensorView.columns,shape]
  have fit:q.column.toNat+11≤(q.output allocated).metadata.columns := parameters_fit q admit allocated allocation
  exact Frames.zero_input q.coordinates.metadata u.metadata gradient fresh u.metadata upValid.2.2.2 fresh.upstreamApart
    r (q.column.toNat+c.val) (by omega) (by have hc:=c.isLt;omega) (by omega) heap

theorem zero_word : Packing.value (fun _ : Fin 4=> (0 : BitVec 8))=FP.Value.zero .fp32 := by rfl

def reference (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r : Nat) (c : Fin 3) : Word :=
  Packing.word (Software.backward .ieeeValue
    (Packing.value (Equivalence.forwardInputs b heap r 0))
    (Packing.value (Equivalence.forwardInputs b heap r 1))
    (Packing.value (Equivalence.forwardInputs b heap r 2))
    (fun j=>Packing.value (Storage.read heap (b.upstream+4*(r*p.upstreamRowStride+(p.columnOffset+j.val)*p.upstreamColStride))))
    (fun _=>FP.Value.zero .fp32) c)

theorem reference_before_allocation (q : Request) (admit : Admitted q) (allocated : TensorObject)
    (allocation : Allocation q allocated) (u gradient : TensorObject)
    (upValid : UpstreamContract q.coordinates.metadata u.metadata (q.output allocated).metadata.columns)
    (fresh : FreshGradient q.coordinates.metadata u.metadata gradient) (heap : Heap)
    (r : Nat) (hr:r<q.coordinates.metadata.rows) (c:Fin 3) :
    let p:=backwardParameters (parameters q admit allocated) u.metadata
    let b:=backwardBuffers (save q allocated) u gradient
    let initial:=zeroGradient gradient.metadata.pointer p.rows heap
    Packing.word (Software.backward .ieeeValue
      (Packing.value (Equivalence.backwardInputs p b initial r 0))
      (Packing.value (Equivalence.backwardInputs p b initial r 1))
      (Packing.value (Equivalence.backwardInputs p b initial r 2))
      (fun j=>Packing.value (Equivalence.backwardInputs p b initial r (3+j.val)))
      (fun j=>Packing.value (Equivalence.backwardInputs p b initial r (14+j.val))) c)=reference p b heap r c := by
  dsimp only
  have coord (j:Fin 3) := coordinates q admit u gradient fresh heap r hr j
  have up (j:Fin 11) := upstream q admit allocated allocation u gradient upValid fresh heap r hr j
  have zero (j:Fin 3) := zeroGradient_read gradient.metadata.pointer q.coordinates.metadata.rows heap r j.val hr j.isLt
  simp only [Equivalence.input_upstream,Equivalence.input_gradient]
  simp only [Equivalence.backwardInputs,show (0:Nat)<3 by omega,show (1:Nat)<3 by omega,
    show (2:Nat)<3 by omega,ite_true,Equivalence.forwardInputs,backwardParameters,parameters,
    backwardBuffers,save,reference]
  have c0:=coord 0
  have c1:=coord 1
  have c2:=coord 2
  norm_num only [Fin.val_zero,Fin.val_one,Fin.val_two] at c0 c1 c2
  rw [c0,c1,c2]
  simp only [up,zero,zero_word]

#axiomsClean coordinates
#axiomsClean upstream
#axiomsClean zero_word
#axiomsClean reference_before_allocation
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Initialized
