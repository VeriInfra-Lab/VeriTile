import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Correct
open VeriTile.Triton FP.Structural KernelInterface
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option autoImplicit false

/-- Actual uncached source with separate in-place Q/K halves, coefficient
windows and GQA leader mask; all IO fields are visible in IO.lean. -/
def uncachedIO (rest : Algebra ℝ) (g : Geometry) (pid0 pid1 : Nat) :=
  RotationIO.logicalUncachedIO rest realOps g pid0 pid1

/-- Actual cached source with signed length/table inputs, indirect cache
outputs and in-place Q. The source's early head return masks every window. -/
def cachedIO (rest : Algebra ℝ) (g : Geometry) (pid0 pid1 : Nat) :=
  RotationIO.logicalCachedIO rest realOps g pid0 pid1

/-- Independent split-half arithmetic, shared by the four Q/K channels. -/
def splitRotation (a b c z : ℝ) (odd : Bool) : ℝ :=
  if odd then a*z+b*c else a*c-b*z

/-- Uncached outputs: Q-first, Q-second, K-first, K-second. Masks in the
shared IO retain the source's token tails and GQA leader branch. -/
def uncachedFormula (g : Geometry) (xs : RotationIO.SplitInputs ℝ (RotationIO.ULane g))
    (key odd : Bool) (i : RotationIO.ULane g) : ℝ :=
  splitRotation (xs.payload key Bool.false i) (xs.payload key Bool.true i)
    (xs.cosine i) (xs.sine i) odd

/-- Cached outputs have the same logical Q/K and half coordinates; the shared
IO places K into the metadata-selected cache and retains the full K frame. -/
def cachedFormula (g : Geometry) (xs : RotationIO.CachedInputs ℝ (RotationIO.CLane g))
    (key odd : Bool) (i : RotationIO.CLane g) : ℝ :=
  splitRotation (xs.payload key Bool.false i) (xs.payload key Bool.true i)
    (xs.cosine i) (xs.sine i) odd

def metadata (g : Geometry) := @RotationIO.LogicalMetadataDomain ℝ (RotationIO.CLane g)

/-- Independent split-half rotation (a*c-b*s, a*s+b*c), with actual source
access safety, all Q/K outputs and exact exterior frame. -/
specification rotary_uncached_correct (rest : Algebra ℝ) (g : Geometry) (pid0 pid1 : Nat)
    (heads : ∀ key,0<g.request.heads key) :
    Spec.Real ((uncachedIO rest g pid0 pid1) ⊨ uncachedFormula g) :=
  (RotationIO.logical_uncached_iff rest g pid0 pid1 (uncachedFormula g) rfl).mpr
    (RotationIO.uncached_real rest g pid0 pid1 heads)

/-- The distinct cached kernel rotates Q and writes the rotated K to the
metadata-selected cache page/slot. Its full frame includes every input K cell;
metadata nonnegativity is the original logical domain, separate from Real data. -/
specification rotary_cached_correct (rest : Algebra ℝ) (g : Geometry) (pid0 pid1 : Nat)
    (token : pid1<g.tokens) :
    Spec.Real ((cachedIO rest g pid0 pid1) ⊨{metadata g} cachedFormula g) :=
  (RotationIO.logical_cached_iff rest g pid0 pid1 (cachedFormula g) rfl).mpr
    (RotationIO.cached_real rest g pid0 pid1 token)

#stmtSurfaceSubset rotary_uncached_correct ⊆
  [Spec.Real,uncachedIO,uncachedFormula,LogicalProgramIO.Implements,FP.Structural.Algebra,
   Geometry,Geometry.request,Request.heads,RotationIO.SplitInputs,RotationIO.ULane]
#stmtSurfaceSubset rotary_cached_correct ⊆
  [Spec.Real,cachedIO,cachedFormula,metadata,LogicalProgramIO.ImplementsOn,FP.Structural.Algebra,
   Geometry,Geometry.tokens,RotationIO.CachedInputs,RotationIO.CLane]
#specNonCircular uncachedFormula avoiding [Kernels.uncached,FP.Structural.exec,Typed.uTarget]
#specNonCircular cachedFormula avoiding [Kernels.cached,FP.Structural.exec,Typed.cReference]
#axiomsClean rotary_uncached_correct
#axiomsClean rotary_cached_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RotaryEmbNopad.Correct
