import bench.tritonbench_g.int8_matmul_quantization.Proofs.ProducerFullMemory
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
open VeriTile.Triton FP.Structural ControlArithmetic
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

abbrev Index (p : Parameters) := TileIndex [1,p.width]
def inputOffset (p : Parameters) (s : State α) (i : Index p) :=
  (s.pids 0 % p.rows)*p.inputRowStride+i.2.1.val*p.inputColumnStride
def outputOffset (p : Parameters) (s : State α) (i : Index p) :=
  (s.pids 0 % p.rows)*p.outputRowStride+i.2.1.val*p.outputColumnStride

def loaded {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) : α :=
  M.fp32Load (if i.2.1.val < p.columns then (s.mem p.input (inputOffset p s i)).read .real
    else M.literal (some .fp32) .real 0)
def maximum {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  if p.columns=0 then M.literal (some .fp32) .real 0 else
    M.binary (some .fp32) .real .max (M.literal (some .fp32) .real 0)
      (M.reduceMax (some .fp32) (shape := [1,p.width]) ⟨1, by simp⟩ Bool.false
        (fun i => M.unary (some .fp32) (.abs .real) (loaded M p s i)) (0,PUnit.unit))
def scale {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : α :=
  M.binary (some .fp32) .real .div (maximum M p s) (M.literal (some .fp32) .real 127)
def quotient {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) : α :=
  M.binary (some .fp32) .real .div (loaded M p s i) (scale M p s)
def rowWritten {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [1,p.width]).foldl (fun t i => if i.2.1.val < p.columns then
    t.write p.output (outputOffset p s i) (.mk .int (integer8 (quotient M p s i)).toInt)
    else t) s
def reference {α : Type} [Inhabited α] (M : Algebra α) (integer8 : α → BitVec 8)
    (p : Parameters) (s : State α) : State α :=
  (rowWritten M integer8 p s).write p.scales (s.pids 0*p.scaleStride)
    (.mk .fp16 (M.cast none .real .fp16 (scale M p s)))



theorem tripCount_covered (p : Parameters) (width : 0<p.width) (covers : p.columns≤p.width) :
    tripCount p=if p.columns=0 then 0 else 1 := by
  by_cases empty : p.columns=0
  · simp [tripCount,empty,Nat.div_eq_of_lt (by omega : p.width-1<p.width)]
  · simp [tripCount,empty,cdiv_one p.columns p.width width (Nat.pos_of_ne_zero empty) covers]

theorem maximum_matches {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (width : 0<p.width) (covers : p.columns≤p.width) :
    MaxPass.maximum M p s (tripCount p)=maximum M p s := by
  rw [tripCount_covered p width covers]
  by_cases empty : p.columns=0
  all_goals simp [empty,maximum,MaxPass.maximum,MaxPass.blockMaximum,MaxPass.loaded,
    MaxPass.inputOffset,loaded,inputOffset]

theorem scale_matches {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (width : 0<p.width) (covers : p.columns≤p.width) : finalScale M p s=scale M p s := by
  unfold finalScale scale
  rw [maximum_matches M p s width covers]

#axiomsClean tripCount_covered
#axiomsClean maximum_matches
#axiomsClean scale_matches
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.KernelProducer
