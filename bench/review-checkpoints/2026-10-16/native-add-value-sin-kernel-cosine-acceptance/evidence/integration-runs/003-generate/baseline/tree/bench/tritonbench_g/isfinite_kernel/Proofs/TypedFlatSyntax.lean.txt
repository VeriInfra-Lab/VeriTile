import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatState

/-! Exhaustive relocation of the compute-facing syntax. Only ordinary `Op`
children use the existing syntax transformer; no compute-to-algorithm
projection is used. These definitions retain unsupported constructs and
their metadata. This leaf does not prove evaluator or execution commutation.

The current `ComputeStmt` has four loop constructors: `forLoop`, `forRange`,
`forRangeDyn`, and `forRangeInt`. Every loop and both branch forms occur
explicitly below. Kernel port metadata follows the ordinary flattening
convention, with the single flat region as both input and output. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton

/-- Translate typed computation while retaining its exact compute dtype. -/
def flattenComputeOp (A : FlatAlloc) :
    {d : ComputeDType} → {sh : TileShape} → ComputeOp d sh → ComputeOp d sh
  | _, _, .alg d e => .alg d (A.flattenOp e)
  | _, _, .const value => .const value
  | _, _, .full sh e => .full sh (flattenComputeOp A e)
  | _, _, .load d mem mask padding =>
      .load d (A.flattenAccess mem) (A.flattenMask mask) padding
  | _, _, .bitcast src dst width e =>
      .bitcast src dst width (flattenComputeOp A e)

/-- Preserve every compute-only wrapper, including classifier precision. -/
def flattenExpr (A : FlatAlloc) :
    {d : AlgDType} → {sh : TileShape} → ComputeExpr d sh → ComputeExpr d sh
  | _, _, .alg e => .alg (A.flattenOp e)
  | _, _, .compute e => .compute (flattenComputeOp A e)
  | _, _, .opaque op => .opaque op
  | _, _, .llrint e => .llrint (flattenExpr A e)
  | _, _, .isFinite precision e => .isFinite precision (flattenExpr A e)

mutual

/-- Translate every statement without discarding effects or control paths. -/
def flattenComputeStmt (A : FlatAlloc) : ComputeStmt → ComputeStmt
  | .alg st => .alg (A.flattenStmt st)
  | .assign d sh name e => .assign d sh name (flattenExpr A e)
  | .store d sh mem value mask =>
      .store d sh (A.flattenAccess mem) (flattenExpr A value) (A.flattenMask mask)
  | .atomicAdd nd sh mem value mask =>
      .atomicAdd nd sh (A.flattenAccess mem) (flattenExpr A value) (A.flattenMask mask)
  | .atomicRMW op d sh mem input extra mask dest =>
      .atomicRMW op d sh (A.flattenAccess mem) (flattenExpr A input)
        (match extra with
         | none => none
         | some e => some (flattenExpr A e))
        (A.flattenMask mask) dest
  | .effectMarker op => .effectMarker op
  | .forLoop idx n body => .forLoop idx n (flattenComputeStmts A body)
  | .forRange idx start stop step body =>
      .forRange idx start stop step (flattenComputeStmts A body)
  | .forRangeDyn idx start stop step body =>
      .forRangeDyn idx (A.flattenOp start) (A.flattenOp stop) (A.flattenOp step)
        (flattenComputeStmts A body)
  | .forRangeInt idx start stop step body =>
      .forRangeInt idx (A.flattenOp start) (A.flattenOp stop) (A.flattenOp step)
        (flattenComputeStmts A body)
  | .ifThen cond body => .ifThen (A.flattenOp cond) (flattenComputeStmts A body)
  | .ifThenElse cond thenBody elseBody =>
      .ifThenElse (A.flattenOp cond)
        (flattenComputeStmts A thenBody) (flattenComputeStmts A elseBody)

/-- Translate every statement in its original order. -/
def flattenComputeStmts (A : FlatAlloc) : List ComputeStmt → List ComputeStmt
  | [] => []
  | st :: rest => flattenComputeStmt A st :: flattenComputeStmts A rest

end

/-- Relocate the complete compute body and give it the flat region metadata. -/
def flattenComputeKernel (A : FlatAlloc) : ComputeKernel → ComputeKernel
  | .mk _inputs _outputs body =>
      .mk [A.flat] [A.flat] (flattenComputeStmts A body)

variable (A : FlatAlloc)

theorem flattenComputeOp_load (d : ComputeDType) {sh : TileShape}
    (mem : MemAccess d.eraseDType sh) (mask : MaskOpt d.eraseDType sh)
    (padding : PaddingOption) :
    flattenComputeOp A (.load d mem mask padding) =
      .load d (A.flattenAccess mem) (A.flattenMask mask) padding := rfl

theorem flattenComputeOp_bitcast (src dst : ComputeDType) {sh : TileShape}
    (width : src.width = dst.width) (e : ComputeOp src sh) :
    flattenComputeOp A (.bitcast src dst width e) =
      .bitcast src dst width (flattenComputeOp A e) := rfl

/-- Classifier precision and both explicit floating casts survive relocation,
as do the load's own precision and padding. No callback law is asserted. -/
theorem flattenExpr_classifier_casts (classifier : FinitePrecision)
    (arithmetic : IndexComputePrecision) {sh : TileShape}
    (mem : MemAccess .real sh) (mask : MaskOpt .real sh) (padding : PaddingOption) :
    flattenExpr A (.isFinite classifier (.alg
      (.castFloat .real .real (.castFloat .real .real
        (.withPrecision .real arithmetic (.load .real mem mask padding)))))) =
      .isFinite classifier (.alg
        (.castFloat .real .real (.castFloat .real .real
          (.withPrecision .real arithmetic
            (.load .real (A.flattenAccess mem) (A.flattenMask mask) padding))))) := by
  have hcast (e : Op .real sh) :
      A.flattenOp (.castFloat .real .real e) =
        .castFloat .real .real (A.flattenOp e) := by
    rw [FlatAlloc.flattenOp.eq_def]
  have hprecision (e : Op .real sh) :
      A.flattenOp (.withPrecision .real arithmetic e) =
        .withPrecision .real arithmetic (A.flattenOp e) := by
    rw [FlatAlloc.flattenOp.eq_def]
  have hload : A.flattenOp (.load .real mem mask padding) =
      .load .real (A.flattenAccess mem) (A.flattenMask mask) padding := by
    cases mem <;> cases mask <;> rw [FlatAlloc.flattenOp.eq_def] <;> rfl
  have hop :
      A.flattenOp
        (.castFloat .real .real (.castFloat .real .real
          (.withPrecision .real arithmetic (.load .real mem mask padding)))) =
        .castFloat .real .real (.castFloat .real .real
          (.withPrecision .real arithmetic
            (.load .real (A.flattenAccess mem) (A.flattenMask mask) padding))) := by
    rw [hcast, hcast, hprecision]
    exact congrArg (fun e : Op .real sh =>
      Op.castFloat .real .real (.castFloat .real .real
        (.withPrecision .real arithmetic e))) hload
  simpa only [flattenExpr] using
    congrArg (fun e : Op .real sh => ComputeExpr.isFinite classifier (ComputeExpr.alg e)) hop

theorem flattenExpr_opaque {d : AlgDType} {sh : TileShape} (op : String) :
    flattenExpr A (ComputeExpr.opaque (dtype := d) (shape := sh) op) =
      ComputeExpr.opaque op := rfl

theorem flattenComputeStmt_effectMarker (op : String) :
    flattenComputeStmt A (.effectMarker op) = .effectMarker op := rfl

/-- A dynamic range retains its index, endpoints, step and both branch arms. -/
theorem flattenComputeStmt_dynamic_branch (idx : RegName)
    (start stop step : Op .nat []) (cond : Op .bool [])
    (thenBody elseBody : List ComputeStmt) :
    flattenComputeStmt A
        (.forRangeDyn idx start stop step [.ifThenElse cond thenBody elseBody]) =
      .forRangeDyn idx (A.flattenOp start) (A.flattenOp stop) (A.flattenOp step)
        [.ifThenElse (A.flattenOp cond)
          (flattenComputeStmts A thenBody) (flattenComputeStmts A elseBody)] := rfl

theorem flattenComputeStmts_length (body : List ComputeStmt) :
    (flattenComputeStmts A body).length = body.length := by
  induction body with
  | nil => rfl
  | cons st rest ih =>
      simp only [flattenComputeStmts, List.length_cons, ih]

/-- The translated metadata is explicitly nonempty even for an empty body. -/
theorem flattenComputeKernel_mk (inputs outputs : List RegionName)
    (body : List ComputeStmt) :
    flattenComputeKernel A (.mk inputs outputs body) =
      .mk [A.flat] [A.flat] (flattenComputeStmts A body) := rfl

theorem flattenComputeKernel_surfaceBody_length (k : ComputeKernel) :
    (flattenComputeKernel A k).surfaceBody.length = k.surfaceBody.length := by
  cases k with
  | mk inputs outputs body =>
      exact flattenComputeStmts_length A body

#axiomsClean flattenComputeOp_load
#axiomsClean flattenComputeOp_bitcast
#axiomsClean flattenExpr_classifier_casts
#axiomsClean flattenExpr_opaque
#axiomsClean flattenComputeStmt_effectMarker
#axiomsClean flattenComputeStmt_dynamic_branch
#axiomsClean flattenComputeStmts_length
#axiomsClean flattenComputeKernel_mk
#axiomsClean flattenComputeKernel_surfaceBody_length

end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
