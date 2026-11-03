import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.BackwardExecution
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Reference
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.Mathematical_TypedReadback

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Execution
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 40000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false

/-- Complete direct forward execution at the checked output width and column offset. -/
def forward {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (forwardKernel p) s = some t} :=
  VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP.execution M (forwardParameters p) s

def coordinateOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.blockSize]) : Nat := i.1.val*3+p.blockSize*3*s.pids 0

def upstreamOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.blockSize]) : Nat := (i.1.val+p.blockSize*s.pids 0)*p.upstreamRowStride+p.columnOffset*p.upstreamColStride

def logicalOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.blockSize]) : Nat :=
  i.1.val*p.outputWidth+p.blockSize*p.outputWidth*s.pids 0+p.columnOffset

def maskedLoad {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (region : RegionName) (address index capacity : Nat) : α :=
  M.fp32Load (if index<capacity then (s.mem region address).read .real else s.undef .real region address)

def load {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (region : RegionName) (offset capacity : Nat) : α :=
  M.fp32Load (if offset<capacity then (s.mem region offset).read .real else s.undef .real region offset)

def backwardValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (c : Fin 3) (i : TileIndex [p.blockSize]) : α :=
  Reference.backwardValues M
    (load M s coordinateRegion (coordinateOffset p s i) (3*p.rows))
    (load M s coordinateRegion (coordinateOffset p s i+1) (3*p.rows))
    (load M s coordinateRegion (coordinateOffset p s i+2) (3*p.rows))
    (fun j=>maskedLoad M s upstreamRegion (upstreamOffset p s i+j.val*p.upstreamColStride)
      (logicalOffset p s i+j.val) (p.outputWidth*p.rows))
    (fun j=>load M s gradientRegion (coordinateOffset p s i+j.val) (3*p.rows)) c

def channelWrite {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (c : Fin 3) (t : State α) : State α :=
  (TileShape.allIndices [p.blockSize]).foldl (fun u i=>
    if coordinateOffset p s i+c.val<3*p.rows then
      u.write gradientRegion (coordinateOffset p s i+c.val) (.mk .real (backwardValue M p s c i)) else u) t

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  (List.finRange 3).foldl (fun t c=>channelWrite M p s c t) s

private theorem fold_memory {α ι : Type} (l : List ι) (r : RegionName)
    (off : ι → Nat) (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (s : State α) :
    (l.foldl (fun t i=>if active i then t.write r (off i) (value i) else t) s).mem =
      l.foldl (fun mem i=>fun r' o=>
        if active i ∧ r'=r ∧ o=off i then value i else mem r' o) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    by_cases hi : active i
    · simp only [List.foldl_cons, if_pos hi, ih]
      congr 1
      funext r' o
      simp [hi, State.write]
    · simp only [List.foldl_cons, if_neg hi, ih]
      congr 1
      funext r' o
      simp [hi]

private theorem cast_add_literal (a b : Nat) :
    (a : Int) + (OfNat.ofNat b : Int) = ((a + b : Nat) : Int) := by rfl
private theorem cast_mul_literal (a b : Nat) :
    (a : Int) * (OfNat.ofNat b : Int) = ((a * b : Nat) : Int) := by rfl
private theorem cast_literal_mul (a b : Nat) :
    (OfNat.ofNat a : Int) * (b : Int) = ((a * b : Nat) : Int) := by rfl

theorem backward_memory {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    (backward M p s).val.mem=(written M p s).mem := by
  simp (config := {maxSteps := 8000000}) [backward, written, channelWrite,
    List.finRange_succ, List.foldl_map, fold_memory, backwardValue, Reference.backwardValues,
    load, maskedLoad, coordinateOffset, upstreamOffset, logicalOffset, bop, Region.cast]
  try rfl
  all_goals simp only [cast_add_literal (b := 1), cast_add_literal (b := 2), cast_add_literal (b := 3), cast_add_literal (b := 4), cast_add_literal (b := 5), cast_add_literal (b := 6), cast_add_literal (b := 7), cast_add_literal (b := 8), cast_add_literal (b := 9), cast_add_literal (b := 10), cast_mul_literal (b := 3), cast_literal_mul (a := 2), cast_literal_mul (a := 3), cast_literal_mul (a := 4), cast_literal_mul (a := 5), cast_literal_mul (a := 6), cast_literal_mul (a := 7), cast_literal_mul (a := 8), cast_literal_mul (a := 9), cast_literal_mul (a := 10),
    ← Nat.cast_add, ← Nat.cast_mul, Int.toNat_natCast, Int.ofNat_lt]

#axiomsClean forward
#axiomsClean backward
#axiomsClean backward_memory
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Execution
