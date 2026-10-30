import bench.repairs.tritonbench_g.mixed_sparse_attention.IO
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOBinding
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.RealReturned

namespace VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole
noncomputable section
set_option maxHeartbeats 2000000

theorem binding (c : Config) (work : List Program) (xs : KernelIO.Values ℝ c)
    (s : State Real.R) (h : (io c work).InputBinding xs s) :
    KernelIO.Reads c (lifted c xs) s :=
  ⟨h (0 : Fin 8),h (1 : Fin 8),h (2 : Fin 8),h (3 : Fin 8),h (4 : Fin 8),
    h (5 : Fin 8),h (6 : Fin 8),h (7 : Fin 8)⟩

def realValues (s : State Real.R) (r : RegionName) (off : Nat) : ℝ :=
  ((s.mem r off).read .real).unbotD 0

theorem cell_of_read (cell : Cell Real.R) (x : ℝ) (h : cell.read .real=some x) :
    cell=Cell.mk .real (some x) := by
  cases cell with
  | mk d v=>cases d <;> simp_all [Cell.read,defaultValue] <;> rfl

theorem real_reads (c : Config) (xs : KernelIO.Values ℝ c) (s : State Real.R)
    (h : KernelIO.Reads c (lifted c xs) s) : Real.Reads c s (realValues s) := by
  intro r hr off
  have finite : ∃ x : ℝ,(s.mem r off).read .real=some x := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl
    · exact ⟨KernelIO.query c xs off,h.query off⟩
    · exact ⟨KernelIO.key c xs off,h.key off⟩
    · exact ⟨KernelIO.value c xs off,h.value off⟩
  obtain ⟨x,hx⟩:=finite
  simpa [realValues,hx,ofFloat] using cell_of_read (s.mem r off) x hx

theorem mathematical_source (c : Config) (xs : KernelIO.Values ℝ c) (s : State Real.R)
    (h : KernelIO.Reads c (lifted c xs) s) (i : TileIndex [c.Z*c.H,c.T,c.BD]) :
    Real.sparseExpected c s (realValues s) i.1.val i.2.1.val i.2.2.1.val=
      mathematical c xs (0 : Fin 1) i := by
  simp [Real.sparseExpected,mathematical,realValues,h.query,h.key,h.value,
    h.length,h.blocks,h.counts,h.offsets,h.columns,KernelIO.query,KernelIO.key,KernelIO.value,
    KernelIO.length,KernelIO.blocks,KernelIO.counts,KernelIO.offsets,KernelIO.columns,lifted,mapValue,KernelIO.inputs]
  all_goals rfl

#axiomsClean real_reads
#axiomsClean mathematical_source
end
end VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
