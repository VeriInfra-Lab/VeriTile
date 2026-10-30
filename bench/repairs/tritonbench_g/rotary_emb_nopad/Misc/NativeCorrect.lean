import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.NativeInterface

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.FPEquiv
open VeriTile.Triton FP.Structural KernelInterface
noncomputable section
set_option autoImplicit false

/-- Actual FP32 source and the original selected primitive model. Logical
payloads retain arbitrary word/provenance inputs; no finite-value guard. -/
def uncachedIO (rest : Algebra Native.Node) (op : Native.Instructions)
    (g : Geometry) (pid0 pid1 : Nat) :=
  RotationIO.uncachedIO rest (Native.ops op Bool.false) g pid0 pid1

def cachedIO (rest : Algebra Native.Node) (op : Native.Instructions)
    (g : Geometry) (pid0 pid1 : Nat) :=
  RotationIO.cachedIO rest (Native.ops op Bool.true) g pid0 pid1

def uncachedReference := RotationIO.uNativeTarget
def cachedReference := RotationIO.cNativeTarget
def metadata := @RotationIO.MetadataDomain Native.Node

/-- Exact source observation against the independent selected FP32 graph,
including operand order and FMA contraction. Primitive instructions are the
model boundary; this task has no historical optimization pair. -/
theorem rotary_uncached_fp32 (rest : Algebra Native.Node) (op : Native.Instructions)
    (g : Geometry) (pid0 pid1 : Nat) (heads : ∀ key,0<g.request.heads key) :
    (uncachedIO rest op g pid0 pid1).Implements (uncachedReference op g) :=
  RotationIO.uncached_native rest op g pid0 pid1 heads

/-- Exact selected native graph for the distinct cached source, retaining all
Q/cache outputs, signed metadata, early return and the full K/exterior frame.
No native=Real claim or numerical transformation premise is introduced. -/
theorem rotary_cached_fp32 (rest : Algebra Native.Node) (op : Native.Instructions)
    (g : Geometry) (pid0 pid1 : Nat) (token : pid1<g.tokens) :
    (cachedIO rest op g pid0 pid1).ImplementsOn (metadata g) (cachedReference op g) :=
  RotationIO.cached_native rest op g pid0 pid1 token

#stmtSurfaceSubset rotary_uncached_fp32 ⊆
  [uncachedIO,uncachedReference,ProgramIO.Implements,FP.Structural.Algebra,Geometry,
   Geometry.request,Request.heads,Native.Node,Native.Instructions,
   VeriTile.Bench.Repairs.RotaryEmb.Typed.instInhabitedNode]
#stmtSurfaceSubset rotary_cached_fp32 ⊆
  [cachedIO,cachedReference,metadata,ProgramIO.ImplementsOn,FP.Structural.Algebra,Geometry,
   Geometry.tokens,Native.Node,Native.Instructions,
   VeriTile.Bench.Repairs.RotaryEmb.Typed.instInhabitedNode]
#specNonCircular uncachedReference avoiding [Kernels.uncached,FP.Structural.exec,Typed.uGraphTarget]
#specNonCircular cachedReference avoiding [Kernels.cached,FP.Structural.exec,Typed.cGraphTarget]
#axiomsClean rotary_uncached_fp32
#axiomsClean rotary_cached_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.FPEquiv
