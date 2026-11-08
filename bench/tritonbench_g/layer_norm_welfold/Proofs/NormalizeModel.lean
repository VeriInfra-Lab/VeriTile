import bench.tritonbench_g.layer_norm_welfold.Proofs.Statistics

/-! Elementwise values and output effects of the literal repaired third loop.
All three FP32 storage reads and both explicit casts are retained. The output
has the original [1,B] shape and its store performs the BF16 conversion. -/
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural
noncomputable section

def normalizeValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row col : Nat) (mean inverse : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 (M.fp32Load ((s.mem X (col+D*row)).read .real)))
  let w := cast32 (cast32 (M.fp32Load ((s.mem W col).read .real)))
  let b := cast32 (cast32 (M.fp32Load ((s.mem Bias col).read .real)))
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) inverse) w) b))

def maskedNormalizeValue {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias : RegionName) (D row col : Nat) (mean inverse : α) : α :=
  let cast32 := M.cast (some .fp32) .real .real
  let x := cast32 (cast32 (M.fp32Load (if col<D then
    (s.mem X (col+D*row)).read .real else s.undef .real X (col+D*row))))
  let w := cast32 (cast32 (M.fp32Load (if col<D then (s.mem W col).read .real else s.undef .real W col)))
  let b := cast32 (cast32 (M.fp32Load (if col<D then (s.mem Bias col).read .real else s.undef .real Bias col)))
  M.cast none .real .bf16 (cast32 (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .sub x mean) inverse) w) b))

def outputChunk {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (X W Bias Y : RegionName) (D B k row : Nat) (mean inverse : α) : State α :=
  (TileShape.allIndices [1,B]).foldl (fun t i =>
    if k*B+i.2.1.val<D then t.write Y (k*B+i.2.1.val+D*row)
      (.mk .bf16 (maskedNormalizeValue M s X W Bias D row (k*B+i.2.1.val) mean inverse))
    else t) s

structure NormalizeRegisters {α : Type} (s t : State α) (B row : Nat) (mean inverse : α) : Prop where
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  mean : t.regs .real [1,1] "tmp6"=some (fun _ => mean)
  rstd : t.regs .real [1,1] "tmp18"=some (fun _ => inverse)
  row : t.regs .nat [1,1] "x0"=some (fun _ => row)
  rbase : t.regs .nat [1,B] "rbase"=some (fun i => i.2.1.val)

structure NormalizeEffect {α : Type} [Inhabited α] (M : Algebra α) (s t : State α)
    (X W Bias Y : RegionName) (D B start count row : Nat) (mean inverse : α) : Prop where
  registers : NormalizeRegisters s t B row mean inverse
  values : ∀ k, start≤k → k<start+count → ∀ i : Fin B, k*B+i.val<D →
    t.mem Y (k*B+i.val+D*row)=.mk .bf16 (normalizeValue M s X W Bias D row (k*B+i.val) mean inverse)
  frame : ∀ q off, (q≠Y ∨ ∀ k, start≤k → k<start+count →
    ∀ i : Fin B, k*B+i.val<D → off≠k*B+i.val+D*row) → t.mem q off=s.mem q off

end
end VeriTile.Bench.Repairs.LayerNormWelfold
