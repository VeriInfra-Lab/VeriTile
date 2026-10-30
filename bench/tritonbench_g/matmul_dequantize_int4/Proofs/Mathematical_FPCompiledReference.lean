import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Mathematical_FPCompiledSource
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
open VeriTile.Triton FP.Structural FPTyped
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
def source (p : Parameters) : ComputeKernel := typed .fp16 p.input p.output p.scale p.weight p.zero
  p.rows p.columns p.inner p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
  p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
  p.zeroGroupStride p.zeroColumnStride p.groupSize p.noGroups p.blockRows p.blockColumns p.blockInner p.groupRows

def prelude (p : Parameters) := (source p).surfaceBody.take 24

def body (p : Parameters) : List ComputeStmt :=
  match (source p).surfaceBody[24]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ code) => code
  | _ => []

def epilogue (p : Parameters) := (source p).surfaceBody.drop 25


/-- Compiled graph dot has an explicit current accumulator operand. -/
structure DotAcc (α : Type) where
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α) → (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (rest : Algebra α) (D : DotAcc α) : Algebra α :=
  {rest with dotAcc := fun precision left right out options {batch} {m} {k} {n} a b c =>
    match left,right,out with
    | .fp16,.fp16,.real => some (D.apply precision options a b c)
    | left,right,out => rest.dotAcc precision left right out options a b c}

/-- Independent initial-memory recurrence, including every original K block. -/
def accumulation {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) : Nat → CIndex p → α
  | 0 => fun _ => M.literal (some .fp32) .real 0
  | step + 1 => D.apply (some .fp32) {} (batch := []) (aValue M p s step)
      (fun j => M.cast (some .fp32) .fp16 .fp16 (bValue M p s step j))
      (accumulation M D p s step)

def outputValue {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) (i : CIndex p) : α :=
  M.cast none .real .fp16 (accumulation M D p s (steps p) i)

theorem graph_boundary (p : Parameters) :
    prelude p = FPTyped.prelude p ∧ epilogue p = FPTyped.epilogue p ∧
    (body p).take 5 = (FPTyped.body p).take 5 ∧
    (body p).drop 6 = (FPTyped.body p).drop 6 := ⟨rfl,rfl,rfl,rfl⟩

theorem carry_dot_node (p : Parameters) :
    (body p)[5]? = some (.assign .real [p.blockRows,p.blockColumns] "accumulator"
      (.compute (.alg .fp32 (.dotAcc .fp16 .fp16 .real (batch := [])
        (.ref .fp16 [p.blockRows,p.blockInner] "a")
        (.castFloat .fp16 .fp16 (.ref .fp16 [p.blockInner,p.blockColumns] "b"))
        (.ref .real [p.blockRows,p.blockColumns] "accumulator") {})))) := by rfl

#axiomsClean graph_boundary
#axiomsClean carry_dot_node
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
