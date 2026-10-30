import bench.repairs.tritonbench_g.uniform_sampling.Proofs.IntegerEncoding
import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Index

namespace VeriTile.Bench.Repairs.UniformSampling.Bounds
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.UniformSampling
open IndexProof

theorem padded_bound (n : Nat) : 4*counterCount n<n+4*block n := by
  have hb:0<block n:=(ReservationProof.block_properties n).2.1
  have h:=Nat.div_mul_le_self (n+4*block n-1) (4*block n)
  have he : 4*counterCount n=((n+4*block n-1)/(4*block n))*(4*block n) := by
    unfold counterCount grid SourceDomain.grid
    ring
  rw [←he] at h
  omega

theorem chunk_nonempty {n start : Nat} (hs:block n∣start) {c : Chunk} (hc:c∈chunks n start) :
    4*c.done<n ∧ 0<chunkSize n c := by
  have hm:=ChunkProof.chunks_properties n start hs hc
  have hp:=padded_bound n
  have hB:=Nat.le_of_dvd hm.2.1 hm.2.2.2.2
  have hn:4*c.done<n:=by nlinarith
  constructor
  · exact hn
  · unfold chunkSize
    omega

theorem native_indices {n start : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) :
    x.2.1*block n+x.2.2.2.val<2^32 ∧
    outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2<2^34 ∧
    chunkSize n x.1≤2^34 := by
  have hm:=ChunkProof.chunks_properties n start hs (schedule_mem.mp hx).1
  have hl:=launch_bounds hs hx
  have hc:x.1.count≤2^32:=by omega
  have hsize:chunkSize n x.1≤4*x.1.count:=Nat.min_le_right _ _
  omega

theorem pointer {n start : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) (scratch : Scratch n) :
    let j:=4*x.1.done+min (outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2) (chunkSize n x.1-1)
    j<n ∧ scratchAddress scratch j+4≤scratch.extent ∧
      scratchAddress scratch j+4≤2^64 := by
  dsimp
  have hn:=chunk_nonempty hs (schedule_mem.mp hx).1
  have hmin:=Nat.min_le_right (outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2) (chunkSize n x.1-1)
  have hsiz:chunkSize n x.1≤n-4*x.1.done:=Nat.min_le_left _ _
  have hb:=scratch.bounded
  have hf:=scratch.finite
  unfold scratchAddress
  omega

/-- Widened native pid/B/lane products and repeated offsets are exact; this is
separate from the ordinary runtime's global launch/serialization contract. -/
theorem widened_machine {n start : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) :
    (BitVec.ofNat 64 (x.2.1*block n+x.2.2.2.val)).toInt=
      ((x.2.1*block n+x.2.2.2.val : Nat):Int) ∧
    (BitVec.ofNat 64 (outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2)).toInt=
      ((outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2 : Nat):Int) := by
  have h:=native_indices hs hx
  exact ⟨i64_exact (by omega),i64_exact (by omega)⟩

theorem actual_integer_expressions {n start : Nat} (hs:block n∣start) {x : Location (block n)}
    (hx:x∈schedule n start) :
    (counterBits x.2.1 (block n) x.2.2.2.val).toInt=((x.2.1*block n+x.2.2.2.val : Nat):Int) ∧
    (outputBits x.2.1 (block n) x.2.2.2.val x.2.2.1.val).toInt=
      ((outputIndex x.2.1 (block n) x.2.2.1 x.2.2.2 : Nat):Int) := by
  rw [counter_bits,output_bits]
  exact widened_machine hs hx

#axiomsClean actual_integer_expressions

#axiomsClean pointer
#axiomsClean widened_machine
#axiomsClean chunk_nonempty
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.UniformSampling.Bounds
