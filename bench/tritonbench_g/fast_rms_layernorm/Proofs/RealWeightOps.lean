import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealWeightProgram

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightOps
open VeriTile.Triton RealWeightProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def colOp (K : Nat) : Op .nat [K] := .ref .nat [K] "col"
def maskOp (K : Nat) : Op .bool [K] := .ref .bool [K] "mask"
def refOp (K : Nat) (name : RegName) : Op .real [K] := .ref .real [K] name
def offsetOp (h : Call) (K : Nat) : Op .nat [K] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "row") (.constNat h.cols)) (colOp K)
def loadOp (h : Call) (K : Nat) (r : RegionName) : Op .real [K] :=
  .load .real (.region r (offsetOp h K)) (.maskOther (maskOp K) ((Op.const (0:ℝ)).broadcast [K]))
def initOps (h : Call) (K : Nat) : List Stmt :=
  [.assign .nat [K] "col" (.add .nat .scalarL (.mul .nat .nil (.programId 0) (.constNat K)) (.arange K)),
   .assign .bool [K] "mask" (.lt .nat .scalarR (colOp K) (.constNat h.cols)),
   .assign .real [K] "acc" (.full [K] (.const 0))]
def body (h : Call) (K : Nat) : List Stmt :=
  [.assign .real [K] "dy" (loadOp h K "dx"),
   .assign .real [K] "x" (loadOp h K "x"),
   .assign .real [] "inv" (.load .real (.region "r" (.ref .nat [] "row")) .none),
   .assign .real [K] "normed" (.mul .real .scalarR (refOp K "x") (.ref .real [] "inv")),
   .ifThen (.constBool (!h.gemma)) [.assign .real [K] "normed" (refOp K "normed")],
   .assign .real [K] "acc" (.add .real (.consSame .nil) (refOp K "acc")
     (.mul .real (.consSame .nil) (refOp K "dy") (refOp K "normed")))]
def tail (K : Nat) : List Stmt :=
  [.store .real [K] (.region "dw" (colOp K)) (refOp K "acc") (.mask (maskOp K))]

theorem bindings (h : Call) (K : Nat) : prefixCode h K=initOps h K ∧ loopBody h K=body h K ∧ tailCode h K=tail K := by
  simp [prefixCode,loopBody,tailCode,source,Kernels.weightBackward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    initOps,body,tail,colOp,refOp,maskOp,loadOp,offsetOp]

#axiomsClean bindings
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealWeightOps
