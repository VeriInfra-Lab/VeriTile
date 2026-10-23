import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.Checked.Success
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8.Proofs.ControlBounds

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
noncomputable section
set_option maxHeartbeats 8000000

theorem signed64_exact (x : Nat) (bound : x < 2^63) : (BitVec.ofNat 64 x).toInt = (x : Int) := by
  have word : (BitVec.ofNat 64 x).toNat = x := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [word]; omega),word]

/-- Each actual signed64 multiply/add and every byte of active unsigned64
pointer formation agrees with the ordinary Tensor backing coordinates. -/
theorem tensor_machine (tensor : Tensor) (bounded : tensor.Bounded) (row column lane : Nat)
    (hr : row < tensor.rows) (hc : column < tensor.columns) (byte : lane < tensor.dtype.bytes) :
    (BitVec.ofNat 64 row*BitVec.ofNat 64 tensor.rowStride + BitVec.ofNat 64 column*BitVec.ofNat 64 tensor.columnStride).toInt =
      (row*tensor.rowStride+column*tensor.columnStride : Nat) ∧
    (tensor.pointer+BitVec.ofNat 64 (tensor.dtype.bytes*(row*tensor.rowStride+column*tensor.columnStride)+lane)).toNat =
      tensor.allocationBase.toNat+tensor.dtype.bytes*(tensor.address row column).2+lane ∧
    tensor.allocationBase.toNat+tensor.dtype.bytes*(tensor.address row column).2+lane < tensor.allocationBase.toNat+tensor.allocationBytes := by
  have products := tensor.pointer_products bounded row column hr hc
  have logical := tensor.logical_bound bounded row column hr hc
  have pointer := bounded.pointer
  simp only [show ¬(tensor.rows = 0 ∨ tensor.columns = 0) by omega,if_false] at pointer
  have index : tensor.pointer.toNat+tensor.dtype.bytes*(row*tensor.rowStride+column*tensor.columnStride)+lane =
      tensor.allocationBase.toNat+tensor.dtype.bytes*(tensor.address row column).2+lane := by
    rw [pointer]
    simp only [Tensor.address,Nat.mul_add]
    omega
  have outside : tensor.allocationBase.toNat+tensor.dtype.bytes*(tensor.address row column).2+lane <
      tensor.allocationBase.toNat+tensor.allocationBytes := by omega
  refine ⟨?_,?_,outside⟩
  · simpa only [BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat] using signed64_exact _ products.2.2.1
  · have bound : tensor.pointer.toNat+(tensor.dtype.bytes*(row*tensor.rowStride+column*tensor.columnStride)+lane) < 2^64 := by
      have physical := bounded.physical
      omega
    have small : tensor.dtype.bytes*(row*tensor.rowStride+column*tensor.columnStride)+lane < 2^64 := by omega
    rw [BitVec.toNat_add,BitVec.toNat_ofNat,Nat.mod_eq_of_lt small,Nat.mod_eq_of_lt bound]
    omega

/-- All four input bytes are tied to the exact physical caller Tensor view,
including storage offsets and zero/broadcast strides. -/
theorem input_bytes (abi : LocalABI Word σ ops) (raw : ByteABI abi backing unit)
    (tensor : Tensor) (binding : TensorBinding backing unit tensor) (bounded : tensor.Bounded)
    (dtype : tensor.dtype = .fp32) (state : σ) (row column lane : Nat)
    (hr : row < tensor.rows) (hc : column < tensor.columns) (byte : lane < 4) :
    raw.observe state tensor.device (tensor.allocationBase.toNat+4*(tensor.address row column).2+lane) =
      wordByte (abi.observe state (tensor.address row column)) lane := by
  have width : unit tensor.storage = 4 := binding.width.trans (by rw [dtype]; rfl)
  have logical := (tensor.logical_bound bounded row column hr hc).1
  rw [dtype] at logical
  have capacity : unit (tensor.address row column).1*((tensor.address row column).2+1) ≤
      (backing (tensor.address row column).1).bytes := by
    change unit tensor.storage*((tensor.address row column).2+1) ≤ (backing tensor.storage).bytes
    rw [width,binding.capacity]
    simpa only [Storage.bytes,Nat.mul_add,Nat.mul_one] using logical
  have codec := raw.codec state (tensor.address row column) capacity lane (by simpa only [Tensor.address,width] using byte)
  simpa only [Tensor.address,Prod.fst,Prod.snd,width,binding.owner,binding.base] using codec

/-- Admission and allocation metadata discharge all formed source-lane
bounds, including clamped masked addresses and the full grouped schedule. -/
theorem checked_formed (input : Host.InputContract live args initial)
    (allocation : Host.Allocation live args.A.rows args.B.columns .fp16 args.A.device initial out ready)
    (hm : 0 < args.A.rows) (hn : 0 < args.B.columns) (hk : 0 < args.A.columns)
    (tile iteration rm cn kl : Nat) (ht : tile < args.problem.tiles) (hi : iteration < args.problem.kTiles)
    (hr : rm < 128) (hc : cn < 256) (hl : kl < 64) :
    args.problem.row tile rm < 2^63 ∧ args.problem.column tile cn < 2^63 ∧ args.problem.k iteration kl < 2^63 ∧
    (args.A.address (args.problem.row tile rm%args.A.rows) (clamp args.A.columns (args.problem.k iteration kl))).2 < 2^63 ∧
    (args.B.address (clamp args.A.columns (args.problem.k iteration kl)) (args.problem.column tile cn%args.B.columns)).2 < 2^63 ∧
    (out.address (clamp args.A.rows (args.problem.row tile rm)) (clamp args.B.columns (args.problem.column tile cn))).2 < 2^63 :=
  formed_addresses args.problem ⟨args.A,args.B,out⟩ (allocation_layout allocation input) hm hn hk tile iteration rm cn kl ht hi hr hc hl

#axiomsClean tensor_machine
#axiomsClean checked_formed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8.Checked
