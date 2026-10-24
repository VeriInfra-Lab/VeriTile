import bench.tritonbench_g.triton_softmax.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax.Precision
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Storage carriers and arithmetic precision are separate: fp64 shares the
real carrier with fp32, but keeps explicit fp64 operation metadata. -/
def channel : Storage → FloatDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5

def arithmetic : Storage → ComputeDType
  | .fp64 => .fp64
  | _ => .fp32

def laneMask (n B : Nat) : Op .bool [B] :=
  .lt .nat .scalarR (.arange B) (.constNat n)

def loadPtr (dtype : TileDType) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] "row_start_ptr") (.arange B))

def paddedLoad (dtype : FloatDType) (n B : Nat) : Op dtype.toTileDType [B] :=
  .load dtype.toTileDType (loadPtr dtype.toTileDType B)
    (.maskOther (laneMask n B)
      (.broadcast (.castFloat .real dtype .negInf) [B]))

def expectedLoad (input : Storage) (n B : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] "row" (.compute (.load .fp32
      (loadPtr .real B) (.maskOther (laneMask n B) (.broadcast .negInf [B]))))
  | .fp64 => .assign .real [B] "row" (.compute (.alg .fp64
      (.load .real (loadPtr .real B) (.maskOther (laneMask n B) (.broadcast .negInf [B])))))
  | .fp16 => .assign .fp16 [B] "row" (.alg (paddedLoad .fp16 n B))
  | .bf16 => .assign .bf16 [B] "row" (.alg (paddedLoad .bf16 n B))
  | .f8e4 => .assign .f8e4 [B] "row" (.alg (paddedLoad .f8e4 n B))
  | .f8e5 => .assign .f8e5 [B] "row" (.alg (paddedLoad .f8e5 n B))

def promotedRow (input : Storage) (B : Nat) : Op .real [B] :=
  match input with
  | .fp32 | .fp64 => .ref .real [B] "row"
  | .fp16 => .castFloat .fp16 .real (.ref .fp16 [B] "row")
  | .bf16 => .castFloat .bf16 .real (.ref .bf16 [B] "row")
  | .f8e4 => .castFloat .f8e4 .real (.ref .f8e4 [B] "row")
  | .f8e5 => .castFloat .f8e5 .real (.ref .f8e5 [B] "row")

def arithmeticExpr (input : Storage) {shape : TileShape} (op : Op .real shape) :
    ComputeExpr .real shape :=
  match input with
  | .fp64 => .compute (.alg .fp64 op)
  | _ => .compute (.alg .fp32 op)

def expectedArithmetic (input : Storage) (B : Nat) : List ComputeStmt :=
  [.assign .real [] "row_max" (arithmeticExpr input
      (.reduceMax ⟨0, by simp⟩ Bool.false (promotedRow input B))),
   .assign .real [B] "numerator" (arithmeticExpr input
      (.exp (.sub .real .scalarR (promotedRow input B) (.ref .real [] "row_max")))),
   .assign .real [] "denominator" (arithmeticExpr input
      (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [B] "numerator"))),
   .assign .real [B] "softmax_output" (arithmeticExpr input
      (.div .real .scalarR (.ref .real [B] "numerator") (.ref .real [] "denominator")))]

/-- Check the actual load dtype, signed-infinity padding, inline pointer
construction and full-tile mask for every source storage pair and candidate. -/
theorem load_precision (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output v out inp si so n B).surfaceBody[3]? =
      some (expectedLoad input n B) := by
  cases input <;> cases output <;> cases v <;> rfl

/-- Max promotes smaller floats; subtraction, exp, sum and division retain the
promoted precision and exact source order, without reassociation or an extra cast. -/
theorem arithmetic_precision (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    ((typedKernel input output v out inp si so n B).surfaceBody.drop 4).take 4 =
      expectedArithmetic input B := by
  cases input <;> cases output <;> cases v <;> rfl

def outputExpr (input output : Storage) (B : Nat) :
    ComputeExpr (channel output).toTileDType [B] :=
  match output with
  | .fp32 => match input with
    | .fp64 => .compute (.alg .fp32 (.castFloat .real .real
        (.withPrecision .real .fp64 (.ref .real [B] "softmax_output"))))
    | _ => .compute (.alg .fp32 (.ref .real [B] "softmax_output"))
  | .fp64 => match input with
    | .fp64 => .compute (.alg .fp64 (.ref .real [B] "softmax_output"))
    | _ => .compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] "softmax_output"))))
  | .fp16 => .alg (.castFloat .real .fp16
      (if input = .fp64 then .withPrecision .real .fp64 (.ref .real [B] "softmax_output")
       else .withPrecision .real .fp32 (.ref .real [B] "softmax_output")))
  | .bf16 => .alg (.castFloat .real .bf16
      (if input = .fp64 then .withPrecision .real .fp64 (.ref .real [B] "softmax_output")
       else .withPrecision .real .fp32 (.ref .real [B] "softmax_output")))
  | .f8e4 => .alg (.castFloat .real .f8e4
      (if input = .fp64 then .withPrecision .real .fp64 (.ref .real [B] "softmax_output")
       else .withPrecision .real .fp32 (.ref .real [B] "softmax_output")))
  | .f8e5 => .alg (.castFloat .real .f8e5
      (if input = .fp64 then .withPrecision .real .fp64 (.ref .real [B] "softmax_output")
       else .withPrecision .real .fp32 (.ref .real [B] "softmax_output")))

def expectedStore (input output : Storage) (n B : Nat) : ComputeStmt :=
  .store (channel output).toTileDType [B]
    (.ptr (.ptrAdd .scalarL (.ref .ptr [] "out_row_start_ptr") (.arange B)))
    (outputExpr input output B) (.mask (laneMask n B))

/-- Exactly the source's destination conversion occurs after division. In
particular fp64 is never silently treated as fp32 because of its real carrier. -/
theorem store_precision (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output v out inp si so n B).surfaceBody[8]? =
      some (expectedStore input output n B) := by
  cases input <;> cases output <;> cases v <;> rfl

/-- The sole candidate difference is before the complete floating suffix. -/
theorem identical_floating_suffix (input output : Storage) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel input output Bool.false out inp si so n B).surfaceBody.drop 2 =
      (typedKernel input output Bool.true out inp si so n B).surfaceBody.drop 2 := by
  cases input <;> cases output <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax.Precision
