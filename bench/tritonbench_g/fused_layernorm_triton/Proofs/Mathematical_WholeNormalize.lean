import bench.tritonbench_g.fused_layernorm_triton.Proofs.Mathematical_WholeWelford

namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
open VeriTile.Triton FP.Structural FPStages
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem source_column_lt (k b j d : Nat) :
    ((k : Int)*(b : Int)+(j : Int) < (d : Int)) ↔ k*b+j<d := by norm_cast

theorem source_address (k b j d r : Nat) :
    ((k : Int)*(b : Int)+(j : Int)+(d : Int)*(r : Int)).toNat = k*b+j+d*r := by
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

theorem source_column_address (k b j : Nat) :
    ((k : Int)*(b : Int)+(j : Int)).toNat = k*b+j := by
  simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

/-- Proof decomposition of the existing candidate's actual scalar views
and second-loop iteration, with the range interpreter's signed counter binding.
Every instruction is projected from the canonical complete source. -/
def normalizeChunk (X W Bias Y : RegionName) (D B k : Nat) : List ComputeStmt :=
  VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.projectionBody X W Bias Y "Mean" "Rstd" 0 D B ++
  [.assign .int [] "roffset" (.alg (.constInt (k*B : Nat)))] ++
  VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.normalizationBody Bool.true X W Bias Y "Mean" "Rstd" 0 D B

def normalizeValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row col : Nat) (mean inverse : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 (M.fp32Load ((s.mem X (col + D * row)).read .real)))
  let w := cast32 (cast32 (M.fp32Load ((s.mem W col).read .real)))
  let b := cast32 (cast32 (M.fp32Load ((s.mem Bias col).read .real)))
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) inverse) w) b))

def maskedNormalizeValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row col : Nat) (mean inverse : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 (M.fp32Load (if col < D then (s.mem X (col + D * row)).read .real else s.undef .real X (col + D * row))))
  let w := cast32 (cast32 (M.fp32Load (if col < D then (s.mem W col).read .real else s.undef .real W col)))
  let b := cast32 (cast32 (M.fp32Load (if col < D then (s.mem Bias col).read .real else s.undef .real Bias col)))
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) inverse) w) b))

def outputChunk {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias Y : RegionName) (D B k row : Nat) (mean inverse : α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if k * B + i.1.val < D then
      t.write Y (k * B + i.1.val + D * row)
        (.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k * B + i.1.val) mean inverse))
    else t) s

structure NormalizeRegisters {α : Type} (s t : State α) (row : Nat)
    (mean inverse : α) : Prop where
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  mean : t.regs .real [1,1] "tmp3" = some (fun _ => mean)
  rstd : t.regs .real [1,1] "tmp10" = some (fun _ => inverse)
  row : t.regs .int [1,1] "x0" = some (fun _ => (row : Int))

theorem typed_scatter_memory_congr {α ι : Type} (l : List ι) (s t : State α)
    (Y : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) (h : s.mem = t.mem) :
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) s).mem =
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) t).mem := by
  induction l generalizing s t with
  | nil => exact h
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split_ifs with hi
    · exact ih _ _ (by simp only [State.write, h])
    · exact ih s t h

theorem typed_scatter_regs {α ι : Type} (l : List ι) (s : State α)
    (Y : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) :
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) s).regs = s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons]; split_ifs <;> simp [ih]

theorem typed_scatter_undef {α ι : Type} (l : List ι) (s : State α)
    (Y : RegionName) (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (value : ι → Cell α) :
    (l.foldl (fun u i => if active i then u.write Y (offset i) (value i) else u) s).undef = s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons]; split_ifs <;> simp [ih]

theorem normalize_chunk_run {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (X W Bias Y : RegionName)
    (D B k row : Nat) (mean inverse : α)
    (hm : s.regs .real [1,1] "tmp3" = some (fun _ => mean))
    (hr : s.regs .real [1,1] "tmp10" = some (fun _ => inverse))
    (hx : s.regs .int [1,1] "x0" = some (fun _ => (row : Int))) :
    ∃ final, run (withEquality M equal32) (normalizeChunk X W Bias Y D B k) s = some final ∧
      final.mem = (outputChunk M s X W Bias Y D B k row mean inverse).mem ∧
      NormalizeRegisters s final row mean inverse := by
  simp only [normalizeChunk, VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.projectionBody, VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePieces.normalizationBody, ite_true, List.append_assoc]
  simp [ run, step, evalExpr, evalComputeOp,
    evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
    numericLt, withEquality, Region.cast, ofFloat, toFloat, hm, hr, hx, store,
    maskedNormalizeValue, outputChunk,source_column_lt,source_address,source_column_address]
  constructor
  · apply typed_scatter_memory_congr
    simp only [State.setReg_mem]
  · constructor <;> simp [typed_scatter_regs, typed_scatter_undef, hm, hr, hx]

#axiomsClean normalize_chunk_run
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole
