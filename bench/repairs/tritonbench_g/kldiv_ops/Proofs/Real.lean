import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Tile
import bench.tritonbench_g.kldiv_ops.Proofs.Math

namespace VeriTile.Bench.Repairs.KldivOps.RealExecution
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section

abbrev loss := VeriTile.Bench.TritonBenchG.KldivOps.FullMath.loss
abbrev rowLoss := VeriTile.Bench.TritonBenchG.KldivOps.FullMath.rowLoss

def backward (flag : Bool) (target : ℝ) : ℝ :=
  if flag then -Real.exp target else -target

/-- Independent target is defined on logical coordinates before any packing,
masking, program IDs or temporary allocations. -/
def total (flag : Bool) (eps : ℝ) (rows cols : Nat) (prediction target : Nat→ℝ) : ℝ :=
  ∑ row : Fin rows, ∑ col : Fin cols,
    loss flag eps (prediction (row.val*cols+col.val)) (target (row.val*cols+col.val))

def forward (flag : Bool) (eps : ℝ) (red rows cols : Nat)
    (prediction target : Nat→ℝ) (i : Nat) : ℝ :=
  if red=0 then loss flag eps (prediction i) (target i)
  else if red=1 then total flag eps rows cols prediction target
  else if red=2 then total flag eps rows cols prediction target/(rows*cols : Nat)
  else total flag eps rows cols prediction target/rows

/-- The standalone backward is precisely the prediction derivative. It does
not infer an unprovided forward reduction or introduce additional scaling. -/
theorem loss_increment (flag : Bool) (eps prediction target delta : ℝ) :
    loss flag eps (prediction+delta) target-loss flag eps prediction target=
      backward flag target*delta := by
  cases flag <;> simp [loss,VeriTile.Bench.TritonBenchG.KldivOps.FullMath.loss,backward] <;> ring

def sumValues {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (values : Values ℝ .real shape) : Values ℝ .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun output => ∑ k : Fin (TileShape.axisDim shape ax),
      values (TileShape.insertAxisIndex shape ax output k)
  · exact fun output => ∑ k : Fin (TileShape.axisDim shape ax),
      values (TileShape.replaceAxisIndex shape ax output k)

def algebra : Algebra ℝ where
  literal := fun _ _ r => r
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y
    | .max => max x y | .pow => 0
  unary := fun _ op x => match op with
    | .log => Real.log x | .exp => Real.exp x | .neg _ => -x | _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ => sumValues

theorem scalar_real (flag : Bool) (eps prediction target : ℝ) :
    Tile.scalar algebra flag eps prediction target=loss flag eps prediction target := by
  cases flag <;> rfl

theorem padded_chunk_real (p : Parameters) (flag : Bool) (eps : ℝ) (row chunk : Nat)
    (s : State ℝ) :
    Tile.chunkSum algebra p flag eps row (chunk*p.block) s=
      VeriTile.Bench.TritonBenchG.KldivOps.FullMath.blockLoss flag eps
        (fun i => (s.mem p.Y (row*p.cols+i)).read .real)
        (fun i => (s.mem p.T (row*p.cols+i)).read .real)
        p.cols p.block chunk := by
  unfold Tile.chunkSum VeriTile.Bench.TritonBenchG.KldivOps.FullMath.blockLoss
  simp only [algebra,sumValues,TileShape.axisDim,TileShape.insertAxisIndex]
  apply Finset.sum_congr rfl
  intro i _
  rw [VeriTile.Bench.TritonBenchG.KldivOps.FullMath.paddedLoss_eq]
  by_cases h : chunk*p.block+i.val<p.cols
  · simp only [Tile.lane,h,if_true]
    exact scalar_real flag eps _ _
  · simp [h,Tile.zero,algebra]

/-- Reuse the existing symbolic block-coverage theorem for every positive
block and arbitrary width, including empty rows and the final partial tile. -/
theorem complete_real_sum (flag : Bool) (eps : ℝ) (prediction target : Nat→ℝ)
    (cols block : Nat) (hb : 0<block) :
    VeriTile.Bench.TritonBenchG.KldivOps.FullMath.prefixLoss flag eps prediction target cols block
      (VeriTile.Bench.TritonBenchG.KldivOps.FullMath.rounds cols block)=
      rowLoss flag eps prediction target cols :=
  VeriTile.Bench.TritonBenchG.KldivOps.FullMath.complete_prefix flag eps prediction target cols block hb

#specNonCircular forward avoiding [Tile.scalar,Tile.chunkSum,algebra]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.RealExecution
