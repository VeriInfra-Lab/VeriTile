import bench.tritonbench_g.int4_matmul.Proofs.FPDotAccSource
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAcc
open VeriTile.Triton FP.Structural FPTyped
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def source (p : Parameters) : ComputeKernel := typed .fp16 p.input p.output p.scale p.weight p.zero
  p.rows p.columns p.inner p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.blockRows p.blockColumns p.blockInner p.groupRows p.steps

def prelude (p : Parameters) := (source p).surfaceBody.take 17

def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[17]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []

def epilogue (p : Parameters) := (source p).surfaceBody.drop 18

/-- The graph primitive retains the current carry. Half storage operands and
fp32 accumulator are fixed by the parsed node; precision/options remain explicit. -/
structure DotAcc (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α) → (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (rest : Algebra α) (D : DotAcc α) : Algebra α :=
  { rest with dotAcc := fun precision left right out options {batch} {m} {k} {n} a b c =>
      match left, right, out with
      | .fp16, .fp16, .real => some (D.apply precision options a b c)
      | left, right, out => rest.dotAcc precision left right out options a b c }

/-- Independent initial-memory, live-carry recurrence for every actual K block. -/
def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) : Nat → CIndex p → α
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step + 1 => D.apply (some .fp32) {} (batch := []) (aValue p s step)
      (fun j => M.cast (some .fp32) .fp16 .fp16 (bValue M p s step j))
      (accumulation M D p s step)

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) (i : CIndex p) : α :=
  M.cast none .real .fp16 (accumulation M D p s p.steps i)

/-- All non-dot statements remain identical to the original source graph. -/
theorem graph_boundary (p : Parameters) :
    prelude p = FPTyped.prelude p ∧ epilogue p = FPTyped.epilogue p ∧
    (body p).take 11 = (FPTyped.body p).take 11 ∧
    (body p).drop 12 = (FPTyped.body p).drop 12 := by
  exact ⟨rfl,rfl,rfl,rfl⟩

theorem carry_dot_node (p : Parameters) :
    (body p)[11]? = some (.assign .real [p.blockRows,p.blockColumns] "accumulator"
      (.compute (.alg .fp32 (.dotAcc .fp16 .fp16 .real (batch := [])
        (.ref .fp16 [p.blockRows,p.blockInner] "a")
        (.castFloat .fp16 .fp16 (.ref .fp16 [p.blockInner,p.blockColumns] "b"))
        (.ref .real [p.blockRows,p.blockColumns] "accumulator") {})))) := by
  rfl

#axiomsClean graph_boundary
#axiomsClean carry_dot_node
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAcc
