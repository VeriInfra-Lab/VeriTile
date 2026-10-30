import bench.repairs.tritonbench_g.int8_dequant_matmul_scalar_cast.IO
import VeriTile.Triton.Float.PerAddressProgramIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOInputReads
open VeriTile Triton FP.Structural
noncomputable section
/-- Only channels read to compute a partial value. C is intentionally absent:
its changing atomic seed is handled separately by the source-cell trace. -/
def DataCells {F H : Type} (s : Shape) (c : Config) (hasBias : Bool)
    (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H)) : Prop :=
  ∀ i j, i.val≠5 → (IOLayout.memory F H s c hasBias).inputMask i vals j →
    initial.mem (IOLayout.inputs s)[i].buf
      (TileShape.linearIndex (IOLayout.inputs s)[i].shape j) =
        Cell.mk (IOLayout.inputs s)[i].dtype (vals i j)

variable {F H : Type} [Inhabited (Sum F H)]
variable (s : Shape) (c : Config) (hasBias : Bool)
variable (vals : (IOLayout.memory F H s c hasBias).Inputs) (initial : State (Sum F H))
variable (cells : DataCells s c hasBias vals initial)
include cells

theorem a (r : Fin s.M) (k : Fin s.K) :
    (initial.mem "A" (r.val*s.K+k.val)).read .int=vals 0 (r,k,PUnit.unit) := by
  have h:=cells 0 (r,k,PUnit.unit) (by change (0:Nat)≠5; decide) (by simp [IOLayout.memory])
  change initial.mem "A" (TileShape.linearIndex [s.M,s.K] (r,k,PUnit.unit))=
    Cell.mk .int (vals 0 (r,k,PUnit.unit)) at h
  simpa [TileShape.linearIndex] using congrArg (fun x=>x.read .int) h

theorem b (k : Fin s.K) (n : Fin s.N) :
    (initial.mem "B" (k.val*s.N+n.val)).read .int=vals 1 (k,n,PUnit.unit) := by
  have h:=cells 1 (k,n,PUnit.unit) (by change (1:Nat)≠5; decide) (by simp [IOLayout.memory])
  change initial.mem "B" (TileShape.linearIndex [s.K,s.N] (k,n,PUnit.unit))=
    Cell.mk .int (vals 1 (k,n,PUnit.unit)) at h
  simpa [TileShape.linearIndex] using congrArg (fun x=>x.read .int) h

theorem rowScale (r : Fin s.M) :
    (initial.mem "state_x" r.val).read .real=vals 2 (r,PUnit.unit) := by
  have h:=cells 2 (r,PUnit.unit) (by change (2:Nat)≠5; decide) (by simp [IOLayout.memory])
  change initial.mem "state_x" (TileShape.linearIndex [s.M] (r,PUnit.unit))=
    Cell.mk .real (vals 2 (r,PUnit.unit)) at h
  simpa [TileShape.linearIndex] using congrArg (fun x=>x.read .real) h

theorem columnScale (n : Fin s.N) :
    (initial.mem "state_w" n.val).read .real=vals 3 (n,PUnit.unit) := by
  have h:=cells 3 (n,PUnit.unit) (by change (3:Nat)≠5; decide) (by simp [IOLayout.memory])
  change initial.mem "state_w" (TileShape.linearIndex [s.N] (n,PUnit.unit))=
    Cell.mk .real (vals 3 (n,PUnit.unit)) at h
  simpa [TileShape.linearIndex] using congrArg (fun x=>x.read .real) h

theorem bias (n : Fin s.N) (enabled : hasBias=Bool.true) :
    (initial.mem "bias" n.val).read .real=vals 4 (n,PUnit.unit) := by
  have h:=cells 4 (n,PUnit.unit) (by change (4:Nat)≠5; decide) (by simp [IOLayout.memory,enabled])
  change initial.mem "bias" (TileShape.linearIndex [s.N] (n,PUnit.unit))=
    Cell.mk .real (vals 4 (n,PUnit.unit)) at h
  simpa [TileShape.linearIndex] using congrArg (fun x=>x.read .real) h

omit cells in
theorem output (allCells : (IOLayout.memory F H s c hasBias).InputCells id vals initial) (r : Fin s.M) (n : Fin s.N) (split : c.S≠1) :
    initial.mem "C" (r.val*s.N+n.val)=Cell.mk .fp16 (vals 5 (r,n,PUnit.unit)) := by
  have h:=allCells 5 (r,n,PUnit.unit) (by simp [IOLayout.memory,split])
  change initial.mem "C" (TileShape.linearIndex [s.M,s.N] (r,n,PUnit.unit))=
    Cell.mk .fp16 (vals 5 (r,n,PUnit.unit)) at h
  simpa [TileShape.linearIndex] using h

#axiomsClean a
#axiomsClean b
#axiomsClean rowScale
#axiomsClean columnScale
#axiomsClean bias
#axiomsClean output
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOInputReads
