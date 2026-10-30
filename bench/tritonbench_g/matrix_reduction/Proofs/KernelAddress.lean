import bench.tritonbench_g.matrix_reduction.Contract
import bench.tritonbench_g.matrix_reduction.Proofs.SourceAddress

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.KernelAddress
open VeriTile.Triton TypedSource

/-- The generated input descriptor uses the same address as the native block
 pointer. Coordinates are extended before i64 stride products; the proof uses
 only successful frontend bounds and ordinary backing storage. Memory-model
 references: documents/GpuMemoryModel.md and fp_source_evidence.json. -/
theorem input_address (d : Storage) (M N : Nat) (h : Native.Admitted M N)
    (x : InputStorage d M N) (i : Fin M) (j : Fin N) :
    Native.inputPointer x.base (storageBytes d) (max N 1) 1 i.val j.val = x.address i j := by
  have bytes : storageBytes d = Native.bytes d := by cases d <;> rfl
  simpa only [bytes,InputStorage.address] using Native.input_address_exact h d x.base
    (by simpa only [←bytes] using x.span) i j

/-- The actual observed data_ptr is the first generated logical element. -/
theorem input_data_pointer (d : Storage) (M N : Nat)
    (x : InputStorage d M N) (i : Fin M) (j : Fin N) :
    (x.metadata.dataPtr + BitVec.ofNat 64 (storageBytes d*(i.val*N+j.val))).toNat =
      x.address i j := by
  rw [x.pointer,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have positive : 0 < storageBytes d := by cases d <;> decide
  have inside := Nat.mul_lt_mul_of_pos_left (Native.linear_index_lt i j) positive
  exact Nat.mod_eq_of_lt (by have := x.span; omega)

#axiomsClean input_address
#axiomsClean input_data_pointer
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixReduction.KernelAddress
