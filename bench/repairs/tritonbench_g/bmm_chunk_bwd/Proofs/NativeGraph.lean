import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.FPDefault

/-! Staged lowering for the actual sm90 TTIR accumulator. A dot payload is an
unmaterialized compiler value, not a rounded floating number. Consuming it in
the immediately following addition produces a native dotAcc graph node. There
is no asserted identity between separately rounded dot/add and fused dotAcc. -/
namespace VeriTile.Bench.Repairs.BmmChunkBwd.NativeGraph
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section

inductive Expr where
  | leaf (word : BitVec 32)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (x : Expr)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (x y : Expr)
  | pendingDot (choice : Fin 9) (precision : Option ComputeDType)
      (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (i : TileIndex (batch++[m,n]))
  | dotAcc (choice : Fin 9) (precision : Option ComputeDType)
      (left right out : FloatDType) (options : DotOptions)
      (batch : TileShape) (m k n : Nat)
      (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
      (acc : Expr) (i : TileIndex (batch++[m,n]))

instance : Inhabited Expr := ⟨.leaf 0⟩

/-- Default FP32 input precision observed in the actual candidate TTIR. -/
def options : DotOptions := { inputPrecision := some "tf32" }

def consume (precision : Option ComputeDType) (dtype : FloatDType) : Binary → Expr → Expr → Expr
  | .add, acc, .pendingDot choice p l r out opts batch m k n a b i =>
      .dotAcc choice p l r out opts batch m k n a b acc i
  | op, a, b => .binary precision dtype op a b

def algebra (rest : Algebra Expr) : Algebra Expr :=
  { rest with literal := Expr.literal
              cast := Expr.cast
              binary := consume }

def dot (choice : Fin 9) : FPDefault.DotPrimitive Expr where
  plain := fun p {batch} {m k n} a b i => .pendingDot choice p .real .real .real options batch m k n a b i
  typed := fun p l r out opts {batch} {m k n} a b i =>
    ofFloat out (.pendingDot choice p l r out opts batch m k n
      (fun j => toFloat l (a j)) (fun j => toFloat r (b j)) i)

/-- A structural staging rule for a pending graph payload. This theorem has no
floating-point interpretation or split-rounding numerical equality. -/
theorem consume_pending (choice : Fin 9) (p : Option ComputeDType) (l r out : FloatDType)
    (opts : DotOptions) (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    consume p out .add acc (.pendingDot choice p l r out opts batch m k n a b i) =
      .dotAcc choice p l r out opts batch m k n a b acc i := rfl

/-- Complete default source execution also succeeds as a staged native graph.
DirectFP.one_readback proves graph readback;
NativeExecution and Direct.nativeMetadata compose native indices and direct views. Native
floating primitive numerical conformance remains a local deferral. -/
theorem source_success (rest : Algebra Expr) (choice : Fin 9)
    (p : RealProgram.Parameters) (initial : State Expr) :
    ∃ final, FP.Structural.exec (FP.ProfiledRewrite.engine (FPDefault.backend (algebra rest) (dot choice)))
      (FPDefault.source (tuning choice).BM (tuning choice).BN (tuning choice).BCS p) initial = some final :=
  FPDefault.source_success (algebra rest) (dot choice) _ _ _ p initial

#axiomsClean consume_pending
#axiomsClean source_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.NativeGraph
