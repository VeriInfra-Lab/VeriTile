import bench.repairs.tritonbench_g.softmax_triton3.Proofs.NativeIO

/-! Auxiliary single-source native correctness, retained with its historical
full declaration names and statements. This is not a two-program equivalence.
The public FPEquiv entry separately proves the actual reciprocal program pair. -/

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxTriton3.NativeGraph
noncomputable section

/-- Independent ordered native graph: source subtraction, optional mask add,
log2(e) multiply, exp2, selected reduction and division. -/
def reference (inst : Instructions) (profile : Profile) := NativeGraph.row inst profile.vector

def negateWord (x : Word) : Word := x ^^^ (BitVec.ofNat 32 (2^31))

/-- State-free native formula on the public IO's logical input channels. -/
abbrev nativeTarget := NativeIO.nativeTarget

/-- The complete row source executes safely and yields the independent native
instruction graph on every FP32 word. IO.lean records its source, windows and
column-owner observation. Primitive instructions remain explicit in `inst`;
no floating transformation law is assumed. -/
theorem softmax_kernel_fp32 (inst : Instructions) (profile : Profile)
    (out inp : RegionName) (mask : Option RegionName) (rows n B p : Nat)
    (positive : 0<B) (cover : n≤B) (valid : p<rows) :
    (softmaxIO inst profile out inp mask rows n B p).Implements
      (nativeTarget inst profile out inp mask n B) :=
  NativeIO.correct inst profile out inp mask rows n B p positive cover valid

#stmtSurfaceSubset softmax_kernel_fp32 ⊆
  [Instructions,Profile,Word,Node,SoftmaxTriton1.NativeGraph.Word,RegionName,
   ProgramIO.Implements,softmaxIO,nativeTarget]
#specNonCircular reference avoiding [softmax_kernel,NativeGraph.evaluate,Floating.evaluate]
#specNonCircular NativeIO.nativeTarget avoiding [softmax_kernel,NativeGraph.evaluate,Floating.evaluate]
#axiomsClean softmax_kernel_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.FPEquiv
