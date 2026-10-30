import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Source

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

def sameMatrix : Broadcast [16,64] [16,64] [16,64] := .consSame (.consSame .nil)
def samePair : Broadcast [16,32] [16,32] [16,32] := .consSame (.consSame .nil)
def fp {sh : TileShape} (e : Op .real sh) : ComputeExpr .real sh := .compute (.alg .fp32 e)
def matrixRef (name : RegName) : Op .real [16,64] := .ref .real [16,64] name
def pairRef (name : RegName) : Op .real [16,32] := .ref .real [16,32] name

def sourceDotNode : ComputeStmt :=
  .assign .real [16,64] "accumulator" (fp (.add .real sameMatrix (matrixRef "accumulator")
    (.dot (batch:=[]) (matrixRef "x") (.ref .real [64,64] "w"))))

def carryDotNode : ComputeStmt :=
  .assign .real [16,64] "accumulator" (fp (.dotAcc .real .real .real (batch:=[])
    (matrixRef "x") (.ref .real [64,64] "w") (matrixRef "accumulator") {}))

def dotIndex (fp8 : Bool) : Nat := if fp8 then 11 else 6

theorem source_dot_site (fp8 outputFP8 : Bool) (p : Arguments) :
    (body fp8 outputFP8 p)[dotIndex fp8]?=some sourceDotNode := by
  cases fp8 <;> cases outputFP8 <;> rfl

theorem source_dot_partition (fp8 outputFP8 : Bool) (p : Arguments) :
    body fp8 outputFP8 p=(body fp8 outputFP8 p).take (dotIndex fp8) ++
      [sourceDotNode] ++ (body fp8 outputFP8 p).drop (dotIndex fp8+1) := by
  cases fp8 <;> cases outputFP8 <;> rfl

def selectedBody (fp8 outputFP8 : Bool) (p : Arguments) : List ComputeStmt :=
  (body fp8 outputFP8 p).take (dotIndex fp8) ++ [carryDotNode] ++
    (body fp8 outputFP8 p).drop (dotIndex fp8+1)

def rotaryBody (fp8 outputFP8 : Bool) (p : Arguments) : List ComputeStmt :=
  match ((epilogue fp8 outputFP8 p)[3]? : Option ComputeStmt) with
  | some (.ifThen _ statements) => statements
  | _ => []

def pairMul (left right : RegName) : Op .real [16,32] :=
  .mul .real samePair (pairRef left) (pairRef right)

def sourceRealNode : ComputeStmt :=
  .assign .real [16,32] "rotated_real" (fp (.sub .real samePair
    (pairMul "real" "cosine") (pairMul "imag" "sine")))
def sourceImagNode : ComputeStmt :=
  .assign .real [16,32] "rotated_imag" (fp (.add .real samePair
    (pairMul "real" "sine") (pairMul "imag" "cosine")))

theorem source_rotary_sites (fp8 outputFP8 : Bool) (p : Arguments) :
    (rotaryBody fp8 outputFP8 p)[11]?=some sourceRealNode ∧
    (rotaryBody fp8 outputFP8 p)[12]?=some sourceImagNode := by
  cases fp8 <;> cases outputFP8 <;>
    simp [rotaryBody,epilogue,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,
      sourceRealNode,sourceImagNode,fp,pairMul,pairRef,samePair,ComputeDType.eraseDType]

/-- Finite site choices retain observed per-callee differences. The retained
HEAD_DIM=4 PTX uses `imag*cos + real*sin` with imag*cos fused for Q, but
`sin*real + cos*imag` with sin*real fused for K. No FMA/outer-add or operand-
commutation equality is assumed for an arbitrary floating carrier. -/
structure RotaryPlan where
  swapFactors : Bool
  fuseRealSin : Bool

def orderedFactors (swap : Bool) (left right : RegName) : RegName × RegName :=
  if swap then (right,left) else (left,right)

def nativeRealNode (plan : RotaryPlan) : ComputeStmt :=
  let first := orderedFactors plan.swapFactors "real" "cosine"
  let second := orderedFactors plan.swapFactors "imag" "sine"
  .assign .real [16,32] "rotated_real" (.fma .fp32 samePair samePair
    (fp (pairRef first.1)) (fp (pairRef first.2))
    (fp (.negFloat .real (pairMul second.1 second.2))))

def nativeImagNode (plan : RotaryPlan) : ComputeStmt :=
  let first := if plan.fuseRealSin then orderedFactors plan.swapFactors "real" "sine"
    else orderedFactors plan.swapFactors "imag" "cosine"
  let second := if plan.fuseRealSin then orderedFactors plan.swapFactors "imag" "cosine"
    else orderedFactors plan.swapFactors "real" "sine"
  .assign .real [16,32] "rotated_imag" (.fma .fp32 samePair samePair
    (fp (pairRef first.1)) (fp (pairRef first.2)) (fp (pairMul second.1 second.2)))

def selectedRotary (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) :=
  (rotaryBody fp8 outputFP8 p).take 11 ++ [nativeRealNode plan,nativeImagNode plan] ++
    (rotaryBody fp8 outputFP8 p).drop 13

theorem source_rotary_partition (fp8 outputFP8 : Bool) (p : Arguments) :
    rotaryBody fp8 outputFP8 p=(rotaryBody fp8 outputFP8 p).take 11 ++
      [sourceRealNode,sourceImagNode] ++ (rotaryBody fp8 outputFP8 p).drop 13 := by
  cases fp8 <;> cases outputFP8 <;>
    simp [rotaryBody,epilogue,source,TypedSource.kernel,TypedSource.halfHalf,TypedSource.halfFP8,TypedSource.fp8Half,TypedSource.fp8FP8,ComputeKernel.surfaceBody,
      sourceRealNode,sourceImagNode,fp,pairMul,pairRef,samePair,ComputeDType.eraseDType]

/-- Selected local syntax changes. There is no claim that these native
expressions equal the uncontracted source graph under arbitrary FP operations. -/
def selectedEpilogue (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) :=
  (epilogue fp8 outputFP8 p).take 3 ++
    [.ifThen (.constBool p.rotary) (selectedRotary plan fp8 outputFP8 p)] ++
    (epilogue fp8 outputFP8 p).drop 4

def selectedSource (plan : RotaryPlan) (fp8 outputFP8 : Bool) (p : Arguments) :=
  prelude fp8 outputFP8 p ++
    [.forRangeDyn "step" (.constNat 0) (countExpr p) (.constNat 1)
      (selectedBody fp8 outputFP8 p)] ++ selectedEpilogue plan fp8 outputFP8 p

#axiomsClean source_dot_site
#axiomsClean source_dot_partition
#axiomsClean source_rotary_sites
#axiomsClean source_rotary_partition
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
