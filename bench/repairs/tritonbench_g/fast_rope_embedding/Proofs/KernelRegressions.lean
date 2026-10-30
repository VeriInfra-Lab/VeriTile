import bench.tritonbench_g.fast_rope_embedding.Kernels
import bench.repairs.tritonbench_g.fast_rope_embedding.Contract
import bench.repairs.tritonbench_g.rope_embedding.Proofs.SourceBinding

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Bench.Repairs.RopeEmbedding

/-- The readable local quote is exactly the original retained typed fast JIT. -/
theorem original_typed_source (c : Call) :
    c.source=VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource.rope
      .fp32 .fp32 .fp32 Bool.false "Q" "C" "S" c.qStride c.cosineStride c.sineStride
      c.geometry.sequence c.geometry.dimension c.geometry.heads c.geometry.block c.backward := rfl

theorem nonunit_not_inverse :
    let forward := Spec.pair Bool.false 1 2 2 0
    Spec.pair Bool.true forward.1 forward.2 2 0≠(1,2) := by
  norm_num [Spec.pair,Prod.mk.injEq]

/-- Odd channels are outside every program's actual write set. -/
theorem odd_tail (c : Call) (row group head : Nat) (hh : head<c.geometry.heads)
    (odd : c.geometry.dimension%2=1) :
    ¬Kernel.Written c.geometry row group "Q" (c.geometry.offset row head (c.geometry.dimension-1)) := by
  apply Kernel.tail_not_written c.geometry row group head (c.geometry.dimension-1) hh
  · omega
  · unfold Request.half; omega

theorem other_program_frame (c : Call) (row group peer head col : Nat)
    (hh : head<c.geometry.heads) (hc : col<c.geometry.dimension)
    (other : peer≠row ∨ head/4≠group) :
    ¬Kernel.Written c.geometry row group "Q" (c.geometry.offset peer head col) :=
  Kernel.other_program_not_written c.geometry row group peer head col hh hc other

/-- Every active store site of the actual strided source belongs to precisely
the footprint used by the word/byte store interpretation. -/
theorem source_store_written (c : Call) (admit : Admitted c) (row group head : Nat)
    (hr : row<c.geometry.rows) (hlo : group*4≤head)
    (hhi : head<min (group*4+4) c.geometry.heads) (second : Bool)
    (lane : Source.Lane c.parameters) (active : lane.1.val<c.geometry.half) :
    Kernel.Written c.geometry row group "Q" (Source.offset c.parameters row head second lane) := by
  have hh := lt_of_lt_of_le hhi (min_le_right _ _)
  have dimension : 0<c.geometry.dimension := by unfold Request.half at active; omega
  have stride := active_stride c admit row hr (Nat.mul_pos (by omega) dimension)
  refine ⟨rfl,head,lane.1.val+if second then c.geometry.half else 0,hlo,hhi,?_,?_⟩
  · cases second <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
  · simp only [Source.offset,Call.parameters]
    rw [stride]
    simp [Request.offset,Request.half,Nat.add_mul,Nat.mul_assoc,Nat.add_assoc]

/-- A concrete formed-offset failure, despite individually small metadata.
The existing Python _geometry guard rejects this shape. -/
theorem source_index_counterexample :
    let r : Request := ⟨1,1,2^30+1,2⟩
    r.offset (2^30) 0 1=2^31+1 ∧ ¬r.offset (2^30) 0 1<2^31 := by decide

#axiomsClean original_typed_source
#axiomsClean nonunit_not_inverse
#axiomsClean odd_tail
#axiomsClean other_program_frame
#axiomsClean source_store_written
#axiomsClean source_index_counterexample
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
