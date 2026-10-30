import bench.tritonbench_g.bmm_chunk_fwd.Proofs.FPInputs

/-! Staged native graph for the actual SM90 accumulating dot. Pending dots are
compiler graph payloads, not separately rounded floating results. Their source
addition consumes the reached carry and forms one native dotAcc node. -/
namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

inductive Expr where
  | leaf (dtype : FloatDType) (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (x y : Expr)
  | pendingDot (config : Host.Config) (precision : Option ComputeDType)
      (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (config : Host.Config) (precision : Option ComputeDType)
      (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (acc : Expr) (i : TileIndex (batch++[m,n]))

instance : Inhabited Expr := ⟨.leaf .fp32 0⟩

/-- Default FP32 input precision observed in the candidate's SM90 TTIR. -/
def options : DotOptions := { inputPrecision := some "tf32" }

def consume (precision : Option ComputeDType) (dtype : FloatDType) : Binary → Expr → Expr → Expr
  | .add, acc, .pendingDot c p l r out opts batch m k n a b i =>
      .dotAcc c p l r out opts batch m k n a b acc i
  | op, a, b => .binary precision dtype op a b

def algebra (rest : Algebra Expr) : Algebra Expr :=
  { rest with literal := Expr.literal, cast := Expr.cast, binary := consume }

/-- `.real` is the DSL compatibility channel for production FP32. Its native
dot node is explicitly FP32/TF32. Typed source branches keep their dtype tags. -/
def backend (rest : Algebra Expr) (c : Host.Config) : FPSuccess.Ops Expr where
  scalar := algebra rest
  dot := fun p {batch} {m k n} a b i =>
    .pendingDot c p .fp32 .fp32 .fp32 options batch m k n a b i
  typedDot := fun p l r out opts {batch} {m k n} a b i =>
    ofFloat out (.pendingDot c p l r out opts batch m k n
      (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i)

/-- Graph staging, with no split-rounding versus fused numerical identity. -/
theorem consume_pending (c : Host.Config) (p : Option ComputeDType) (l r out : FloatDType)
    (opts : DotOptions) (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    consume p out .add acc (.pendingDot c p l r out opts batch m k n a b i) =
      .dotAcc c p l r out opts batch m k n a b acc i := rfl

#axiomsClean consume_pending
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeGraph

#guard_msgs (drop info) in
#auditModuleAxioms
