import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Host
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.FP

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Equivalence
open VeriTile.Triton FP.Structural Packing Storage
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 64000

/-- Loads transport the four source bytes without numerical conversion. -/
def exactLoads (actual : Algebra Word) : Algebra Word := {actual with fp32Load:=id}

def forwardInputs (b : Physical.Buffers) (heap : Heap) (r i : Nat) : Word :=
  Storage.read heap (b.coordinates+4*(r*3+i))

def backwardInputs (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r i : Nat) : Word :=
  if i<3 then forwardInputs b heap r i
  else if i<14 then Storage.read heap (b.upstream+4*(r*p.upstreamRowStride+(p.columnOffset+(i-3))*p.upstreamColStride))
  else Storage.read heap (b.destination+4*(r*3+(i-14)))

theorem input_upstream (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r : Nat) (c : Fin 11) :
    backwardInputs p b heap r (3+c.val)=
      Storage.read heap (b.upstream+4*(r*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride)) := by
  have hc:=c.isLt
  simp [backwardInputs,show ¬3+c.val<3 by omega,show 3+c.val<14 by omega]

theorem input_gradient (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r : Nat) (c : Fin 3) :
    backwardInputs p b heap r (14+c.val)=Storage.read heap (b.destination+4*(r*3+c.val)) := by
  simp [backwardInputs,show ¬14+c.val<3 by omega,show ¬14+c.val<14 by omega]

theorem forward_value (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r : Nat) (c : Fin 11)
    (localLaws : FPProof.ForwardNumerics (exactLoads actual) rest softwareRest cfg (forwardInputs b heap r) c) :
    Grid.value (exactLoads actual) .forward p (Physical.backed .forward p b heap) r c=
      Packing.word (Software.forward cfg (Packing.value (forwardInputs b heap r 0))
        (Packing.value (forwardInputs b heap r 1)) (Packing.value (forwardInputs b heap r 2)) c) := by
  have scalar:=FPProof.scalar_forward (exactLoads actual) rest softwareRest cfg (forwardInputs b heap r) c localLaws
  simpa only [Grid.value,Physical.backed,coordinateRegion,Cell.read_same,exactLoads,forwardInputs,ite_true] using scalar

theorem backward_value (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r : Nat) (c : Fin 3)
    (localLaws : FPProof.BackwardNumerics (exactLoads actual) rest softwareRest cfg (backwardInputs p b heap r) c) :
    Grid.value (exactLoads actual) .backward p (Physical.backed .backward p b heap) r c=
      Packing.word (Software.backward cfg (Packing.value (backwardInputs p b heap r 0))
        (Packing.value (backwardInputs p b heap r 1)) (Packing.value (backwardInputs p b heap r 2))
        (fun j=>Packing.value (backwardInputs p b heap r (3+j.val)))
        (fun j=>Packing.value (backwardInputs p b heap r (14+j.val))) c) := by
  have scalar:=FPProof.scalar_backward (exactLoads actual) rest softwareRest cfg (backwardInputs p b heap r) c localLaws
  simp only [input_upstream,input_gradient] at scalar
  simp only [backwardInputs,show (0:Nat)<3 by omega,show (1:Nat)<3 by omega,
    show (2:Nat)<3 by omega,ite_true,forwardInputs] at scalar
  simpa only [Grid.value,Readback.backwardExpected,Physical.backed,coordinateRegion,upstreamRegion,
    gradientRegion,Cell.read_same,exactLoads,ite_true,ite_false] using scalar

theorem forward (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth)
    (localLaws : ∀ r,r<p.rows → ∀ c,FPProof.ForwardNumerics (exactLoads actual) rest softwareRest cfg (forwardInputs b heap r) c) :
    (∀ r,r<p.rows → ∀ c : Fin 11,∀ k : Fin 4,
      (Physical.launch (exactLoads actual) .forward p b heap).bytes
        (b.destination+4*(r*p.outputWidth+p.columnOffset+c.val)+k.val)=
      Packing.word (Software.forward cfg (Packing.value (forwardInputs b heap r 0))
        (Packing.value (forwardInputs b heap r 1)) (Packing.value (forwardInputs b heap r 2)) c) k) ∧
    (∀ q,(∀ r,r<p.rows → ∀ c : Fin 11,∀ k : Fin 4,q≠b.destination+4*(r*p.outputWidth+p.columnOffset+c.val)+k.val) →
      (Physical.launch (exactLoads actual) .forward p b heap).bytes q=heap q) := by
  have complete:=Physical.complete (exactLoads actual) .forward p b heap fit
  refine ⟨?_,complete.2.2⟩
  intro r hr c k
  exact (complete.2.1 r hr c k).trans (congrFun (forward_value actual rest softwareRest cfg p b heap r c (localLaws r hr c)) k)

theorem backward (actual rest : Algebra Word) (softwareRest : Algebra (FP.Value .fp32))
    (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth)
    (localLaws : ∀ r,r<p.rows → ∀ c,FPProof.BackwardNumerics (exactLoads actual) rest softwareRest cfg (backwardInputs p b heap r) c) :
    (∀ r,r<p.rows → ∀ c : Fin 3,∀ k : Fin 4,
      (Physical.launch (exactLoads actual) .backward p b heap).bytes (b.destination+4*(r*3+c.val)+k.val)=
      Packing.word (Software.backward cfg (Packing.value (backwardInputs p b heap r 0))
        (Packing.value (backwardInputs p b heap r 1)) (Packing.value (backwardInputs p b heap r 2))
        (fun j=>Packing.value (backwardInputs p b heap r (3+j.val)))
        (fun j=>Packing.value (backwardInputs p b heap r (14+j.val))) c) k) ∧
    (∀ q,(∀ r,r<p.rows → ∀ c : Fin 3,∀ k : Fin 4,q≠b.destination+4*(r*3+c.val)+k.val) →
      (Physical.launch (exactLoads actual) .backward p b heap).bytes q=heap q) := by
  have complete:=Physical.complete (exactLoads actual) .backward p b heap fit
  refine ⟨?_,complete.2.2⟩
  intro r hr c k
  exact (complete.2.1 r hr c k).trans (congrFun (backward_value actual rest softwareRest cfg p b heap r c (localLaws r hr c)) k)

#axiomsClean forward
#axiomsClean backward
end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Equivalence
