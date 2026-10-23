import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.IO
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.QuoteOperands
import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.Proofs.FiniteFormula
import VeriTile.Triton.Float.PerAddressProgramIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOBinding
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000

/-- Finite views of the original physical storage. Aliases and offsets are
read exactly as before; each source cell is the same decoded typed view. -/
def values (ops : Ops I F H) (s : Shape) (b : Buffers)
    (m : Int8DequantMatmul.Memory F H) : ChannelValues (Sum F H) (IOLayout.inputs s) :=
  Fin.cases (fun j => (read8 m (b.a,b.aOffset+j.1.val*s.K+j.2.1.val)).toInt)
    (Fin.cases (fun j => (read8 m (b.b,b.bOffset+j.1.val*s.N+j.2.1.val)).toInt)
      (Fin.cases (fun j => Sum.inl (read32 ops m (b.x,b.xOffset+j.1.val)))
        (Fin.cases (fun j => Sum.inl (read32 ops m (b.w,b.wOffset+j.1.val)))
          (Fin.cases (fun j => Sum.inl (read32 ops m (b.bias,b.biasOffset+j.1.val)))
            (Fin.cases (fun j => Sum.inr (read16 ops m (b.out,b.outOffset+j.1.val*s.N+j.2.1.val)))
              (fun i => Fin.elim0 i))))))

theorem data_values (ops : Ops I F H) (s : Shape) (b : Buffers) (m : Int8DequantMatmul.Memory F H) :
    IOLayout.data ops s (values ops s b m) = FiniteFormula.input ops s b m := by
  unfold IOLayout.data FiniteFormula.input
  congr 1
  · funext r k
    exact BitVec.ofInt_toInt
  · funext k n
    exact BitVec.ofInt_toInt

theorem zero (ops : Ops I F H) (s : Shape) (c : Config) (b : Buffers)
    (m : Int8DequantMatmul.Memory F H) (zeroed : Zeroed ops s c b m) :
    IOLayout.domain ops s c (values ops s b m) := by
  intro split r n
  change Sum.inr (read16 ops m (outputAddress s b (r,n))) = Sum.inr ops.hzero
  rw [zeroed_read zeroed split]

theorem source_cells (q : Quote.IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (hasBias : Bool) (b : Buffers) (m : Int8DequantMatmul.Memory F H)
    (divisor : F) (pid split : Nat) :
    (IOLayout.memory F H s c hasBias).InputCells id (values q.ops s b m)
      (Quote.initialState q b m divisor pid split) := by
  intro i j _
  fin_cases i
  · rcases j with ⟨r,k,u⟩; cases u
    change Quote.sourceMemory q b m "A" (TileShape.linearIndex [s.M,s.K] (r,k,PUnit.unit)) =
      Cell.mk .int (read8 m (b.a,b.aOffset+r.val*s.K+k.val)).toInt
    simp [Quote.sourceMemory,TileShape.linearIndex,Nat.add_assoc]
  · rcases j with ⟨k,n,u⟩; cases u
    change Quote.sourceMemory q b m "B" (TileShape.linearIndex [s.K,s.N] (k,n,PUnit.unit)) =
      Cell.mk .int (read8 m (b.b,b.bOffset+k.val*s.N+n.val)).toInt
    simp [Quote.sourceMemory,TileShape.linearIndex,Nat.add_assoc]
  · rcases j with ⟨r,u⟩; cases u
    change Quote.sourceMemory q b m "state_x" (TileShape.linearIndex [s.M] (r,PUnit.unit)) =
      Cell.mk .real (Sum.inl (read32 q.ops m (b.x,b.xOffset+r.val)))
    simp [Quote.sourceMemory,TileShape.linearIndex]
  · rcases j with ⟨n,u⟩; cases u
    change Quote.sourceMemory q b m "state_w" (TileShape.linearIndex [s.N] (n,PUnit.unit)) =
      Cell.mk .real (Sum.inl (read32 q.ops m (b.w,b.wOffset+n.val)))
    simp [Quote.sourceMemory,TileShape.linearIndex]
  · rcases j with ⟨n,u⟩; cases u
    change Quote.sourceMemory q b m "bias" (TileShape.linearIndex [s.N] (n,PUnit.unit)) =
      Cell.mk .real (Sum.inl (read32 q.ops m (b.bias,b.biasOffset+n.val)))
    simp [Quote.sourceMemory,TileShape.linearIndex]
  · rcases j with ⟨r,n,u⟩; cases u
    change Quote.sourceMemory q b m "C" (TileShape.linearIndex [s.M,s.N] (r,n,PUnit.unit)) =
      Cell.mk .fp16 (Sum.inr (read16 q.ops m (b.out,b.outOffset+r.val*s.N+n.val)))
    simp [Quote.sourceMemory,TileShape.linearIndex,Nat.add_assoc]

#axiomsClean data_values
#axiomsClean zero
#axiomsClean source_cells
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOBinding
