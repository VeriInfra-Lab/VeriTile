import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Contract

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
open TileModel
noncomputable section
set_option maxHeartbeats 8000000

/-- Existing Python checks supply exactly the same index envelope for the
candidate. No extra K, stride, or numerical-result restriction is added. -/
theorem admitted_indices (request : Request) (aPointer bPointer out : Nat) (before : Memory Word32)
    (admitted : InputContract request aPointer bPointer before) :
    SourceBounds request.shape (hostBuffers request out) :=
  (VeriTile.Bench.Repairs.Int8MatmulKernel.indexChecks_iff request out).mp
    ((VeriTile.Bench.Repairs.Int8MatmulKernel.indexChecks_iff request 0).mpr admitted.indices)

theorem source_a_address (request : Request) (out : Nat) (before : Memory Word32) (c : Fin 15)
    (legal : request.shape.legal c) (state : State Word32) (f j : Nat)
    (hf : f < 4) (hj : j < request.shape.steps c)
    (i : AIndex (parameters (layout request) request.shape c)) :
    aWord (parameters (layout request) request.shape c)
      { state with mem := (encodedState request out before).mem } f j i =
      before (aAddress (hostBuffers request out)
        (inputRow (parameters (layout request) request.shape c) state i.1)
        ((f*request.shape.steps c+j)*(configs c).bk+i.2.1.val)) := by
  rw [aWord_stream (layout request) request.shape c legal _ f j hf hj]
  exact input_encoding request out before _ _

theorem source_a_index_bound (request : Request) (aPointer bPointer out : Nat) (before : Memory Word32)
    (admitted : InputContract request aPointer bPointer before) (c : Fin 15) (legal : request.shape.legal c)
    (state : State Word32) (mpos : 0 < request.shape.m) (f j : Nat)
    (advance : f*request.shape.steps c+j ≤ 4*request.shape.steps c)
    (i : AIndex (parameters (layout request) request.shape c)) :
    aOffset (parameters (layout request) request.shape c) state f j i < 2^31 := by
  have bound := formed_a request.shape (hostBuffers request out) c
    (admitted_indices request aPointer bPointer out before admitted) legal
    (inputRow (parameters (layout request) request.shape c) state i.1)
    (f*request.shape.steps c+j) i.2.1.val (Nat.mod_lt _ mpos) advance i.2.1.isLt
  simpa [aOffset,parameters,layout,hostBuffers,Parameters.steps,Shape.steps,ceilDiv,
    Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using bound

theorem source_b_index_bound (request : Request) (aPointer bPointer out : Nat) (before : Memory Word32)
    (admitted : InputContract request aPointer bPointer before) (c : Fin 15) (legal : request.shape.legal c)
    (state : State Word32) (npos : 0 < request.shape.n) (j : Nat) (advance : j ≤ request.shape.steps c)
    (i : BIndex (parameters (layout request) request.shape c)) :
    bOffset (parameters (layout request) request.shape c) state j i < 2^31 := by
  have bound := formed_b request.shape (hostBuffers request out) c
    (admitted_indices request aPointer bPointer out before admitted) legal
    (inputColumn (parameters (layout request) request.shape c) state i.2.1)
    j i.1.val (Nat.mod_lt _ npos) advance i.1.isLt
  simpa [bOffset,parameters,layout,hostBuffers,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc,
    Nat.add_comm,Nat.add_left_comm] using bound

/-- The ordinary allocation-base/data-pointer identities plus unchanged
input/output checks provide the original signed64 formed-pointer bounds. -/
def checkedPointers {request : Request} {out aPointer bPointer : Nat}
    (ordinary : VeriTile.Bench.Repairs.Int8MatmulKernel.OrdinaryPointers request out aPointer bPointer)
    (before : Memory Word32) (admitted : InputContract request aPointer bPointer before)
    (cPointer : Nat) (output : VeriTile.Bench.Repairs.Int8MatmulKernel.outputChecks request cPointer = Bool.true) :
    PointerBounds request.shape (hostBuffers request out) :=
  ordinary.checked before admitted cPointer output

theorem active_allocation_coverage (request : Request) (out : Nat) (domain : request.Domain)
    (c : Fin 15) (legal : request.shape.legal c) (state : State Word32)
    (mpos : 0 < request.shape.m) (npos : 0 < request.shape.n)
    (f j : Nat) (hf : f < 4) (hj : j < request.shape.steps c)
    (ar : Fin (configs c).bm) (bn : Fin (configs c).bn) (lane : Fin (configs c).bk) :
    (aAddress (hostBuffers request out)
      (inputRow (parameters (layout request) request.shape c) state ar)
      ((f*request.shape.steps c+j)*(configs c).bk+lane.val)).2 < request.a.capacity ∧
    (bAddress (hostBuffers request out) (j*(configs c).bk+lane.val)
      (inputColumn (parameters (layout request) request.shape c) state bn)).2 < request.b.capacity := by
  have masks := legal_all_masks request.shape c legal f j lane.val hf hj lane.isLt
  exact input_allocation_coverage request out domain _ _ _ _ (Nat.mod_lt _ mpos)
    masks.2.2.1 masks.2.2.2 (Nat.mod_lt _ npos)

#axiomsClean admitted_indices
#axiomsClean source_a_address
#axiomsClean source_a_index_bound
#axiomsClean source_b_index_bound
#axiomsClean checkedPointers
#axiomsClean active_allocation_coverage
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TensorModel
