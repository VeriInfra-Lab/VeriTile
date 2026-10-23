import bench.repairs.tritonbench_g.rmsnorm_implementation.Proofs.NativePrimitives

namespace VeriTile.Bench.Repairs.RmsnormImplementation.NativeInterface
open VeriTile.Triton NativeKernel
open VeriTile.Bench.TritonBenchG.RmsnormImplementation.Whole
noncomputable section

/-- Raw memory addressing only; no Tensor/API metadata is retained. -/
structure View where
  allocation : Nat
  base : Nat
  stride : Nat → Nat

structure Geometry where
  batches : Nat
  rows : Nat
  width : Nat
  input : View
  weights : View
  output : View
  eps : ℝ

def inputOffset (g : Geometry) (b m k : Nat) : Nat :=
  g.input.base+b*g.input.stride 0+m*g.input.stride 1+k*g.input.stride 2
def weightOffset (g : Geometry) (k : Nat) : Nat := g.weights.base+k*g.weights.stride 0
def outputOffset (g : Geometry) (b m k : Nat) : Nat :=
  g.output.base+b*g.output.stride 0+m*g.output.stride 1+k*g.output.stride 2

def OutputUnique (g : Geometry) : Prop :=
  ∀ b m k b' m' k',b<g.batches → m<g.rows → k<g.width →
    b'<g.batches → m'<g.rows → k'<g.width →
    outputOffset g b m k=outputOffset g b' m' k' → b=b' ∧ m=m' ∧ k=k'

/-- Adapter to the retained raw two-pass graph. Unused descriptor metadata is
fixed, not supplied by a Tensor caller or hidden in the kernel premises. -/
def parameters (g : Geometry) : Parameters where
  batches := g.batches
  rows := g.rows
  width := g.width
  eps := g.eps
  input := ⟨[],g.input.stride,g.input.allocation,g.input.base,0,.fp32⟩
  weights := ⟨[],g.weights.stride,g.weights.allocation,g.weights.base,0,.fp32⟩
  output := ⟨[],g.output.stride,g.output.allocation,g.output.base,0,.fp32⟩

def run (ops : Primitives) (red : Reduction) (g : Geometry) (b m : Nat)
    (undefinedWeight : Nat → Word) (before : Memory Word) : Memory Word :=
  executeRow (numerics ops) red (parameters g) b m undefinedWeight before

/-- Complete observed native two-pass execution, independent recurrence and
physical frame. Source/compiler correspondence remains the existing explicit
boundary; the typed source equivalence is stated separately in FPEquiv. -/
theorem observed_kernel (ops : Primitives) (red : Reduction) (g : Geometry)
    (separate : g.input.allocation≠g.output.allocation ∧ g.weights.allocation≠g.output.allocation)
    (unique : OutputUnique g) (b m : Nat) (hb : b<g.batches) (hm : m<g.rows)
    (before : Memory Word) (undefinedWeight : Nat → Word) :
    ∃ final,final=run ops red g b m undefinedWeight before ∧
      (∀ k,k<g.width → final g.output.allocation (outputOffset g b m k)=
        reference ops red g.width g.eps
          (fun j => before g.input.allocation (inputOffset g b m j))
          (fun j => before g.weights.allocation (weightOffset g j)) k) ∧
      (∀ allocation offset,(∀ k,k<g.width →
        ¬(allocation=g.output.allocation ∧ offset=outputOffset g b m k)) →
        final allocation offset=before allocation offset) := by
  refine ⟨_,rfl,?_,?_⟩
  · intro k hk
    exact row_value (numerics ops) red (parameters g) ⟨separate.1,separate.2⟩ unique
      b m hb hm undefinedWeight before k hk
  · exact row_frame (numerics ops) red (parameters g) b m undefinedWeight before

#axiomsClean observed_kernel
#specNonCircular reference avoiding [run,executeRow,outputPass,varianceStep]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.NativeInterface
