import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectAPIs
import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.DirectReal

namespace VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
open Host Helpers HelperHost Physical
noncomputable section

/-- Ordinary installed-library metadata. Literal operand/output strides and the
launch device are filled from the checked tensors, not supplied independently. -/
structure LibraryBuild where
  libraryBuild : String
  algorithm : Nat
  architecture : Nat
  workspaceBytes : Nat
  reducedPrecisionReduction : Bool

def libraryMetadata (p : S1Info) (a : Host.Allocations (arguments p)) (build : LibraryBuild) : Library.Metadata :=
  ⟨build.libraryBuild,build.algorithm,p.input.tensor.device,build.architecture,build.workspaceBytes,
    build.reducedPrecisionReduction,(p.input.tensor.strides.row,p.input.tensor.strides.column),
    (a.scratch.strides.row,a.scratch.strides.column),
    (((arguments p).output a.freshOutput).strides.row,((arguments p).output a.freshOutput).strides.column)⟩
def libraryPlan (implementation : Library.Implementation) (p : S1Info)
    (a : Host.Allocations (arguments p)) (build : LibraryBuild) : Library.Selected (arguments p).packed.shape :=
  implementation.select (arguments p).packed.shape (libraryMetadata p a build)

/-- Per-compilation ISA and physical-register/lane interpretation. Its domain
contains shape/configuration metadata only, never input payloads or output values.
The actual compiler mapping is an explicit external implementation boundary. -/
structure FusedCompilation where
  instruction : Library.MMAKind
  fusion : Nat → Nat → Fusion
structure Compiler where
  compile : Shape → Bool → Fin 17 → FusedCompilation
def compiledChoice (compiler : Compiler) (p : RawInfo) (config : Fin 17) : RawChoice :=
  let selected := compiler.compile (rawShape p) p.noGroups config
  ⟨config,selected.instruction,selected.fusion⟩
def compiledTrials (compiler : Compiler) (p : RawInfo) (choices : List (Fin 17)) := choices.map (compiledChoice compiler p)

def boundS1 (implementation : Library.Implementation) (build : LibraryBuild) (p : S1Info)
    (a : Host.Allocations (arguments p)) (I : Native.Instructions) (choices : List (Fin 8)) (chosen : Fin 8)
    (heap : Heap) (caller : Nat) := s1Wrapper p a I (libraryPlan implementation p a build) choices chosen heap caller

def boundRaw (compiler : Compiler) (p : RawInfo) (h : RawAccepted p) (I : Native.Instructions)
    (choices : List (Fin 17)) (chosen : Fin 17) (heap : Heap) (caller : Nat) :=
  rawCall p h I (compiledTrials compiler p choices) (compiledChoice compiler p chosen) heap caller

structure BoundNative (actual reference : Native.Instructions) : Prop where
  checked : CheckedNative actual reference
  s1 : ∀ implementation build p a choices chosen heap caller result,
    boundS1 implementation build p a actual choices chosen heap caller=.ok result →
    S1Claim (arguments p) a reference (libraryPlan implementation p a build) heap caller result ∧ LaunchFacts (arguments p).packed ∧ S1Accepted p
  fusedRaw : ∀ compiler p (h : RawAccepted p) choices chosen heap caller,
    RawClaim p reference (compiledChoice compiler p chosen) heap caller (boundRaw compiler p h actual choices chosen heap caller)

theorem bound_native (I : Native.Instructions) : BoundNative I I := by
  refine ⟨checked_native I,?_,?_⟩
  · intro implementation build p a choices chosen heap caller result returned
    exact s1_returned p a I (libraryPlan implementation p a build) choices chosen heap caller result returned
  · intro compiler p h choices chosen heap caller
    exact raw_native p h I (compiledTrials compiler p choices) (compiledChoice compiler p chosen) heap caller

theorem bound_if_local (actual reference : Native.Instructions) (numeric : Native.Correspondence actual reference) :
    BoundNative actual reference := by
  rw [←Native.correspondence_eq actual reference numeric]
  exact bound_native actual

structure BoundReal : Prop where
  dequant : ∀ p dtype device,checkDequant p dtype device=.ok () → ∀ a choices chosen initial caller,
    RealDequantClaim p dtype a initial caller (realDequantCall p dtype a choices chosen initial caller)
  s1 : ∀ implementation build p,checkS1 p=.ok () → ∀ a choices chosen initial caller,
    RealS1Claim p a initial caller (realS1Call p a (libraryPlan implementation p a build) choices chosen initial caller)
  quant : ∀ p dtype,(dtype=.fp16 ∨ dtype=.fp32) → QuantMetadata p dtype → ∀ a order initial caller,
    RealQuantClaim p dtype a order initial caller (realQuantCall p dtype a order initial caller)
  fusedRaw : ∀ compiler p (h : RawAccepted p) choices chosen initial caller,
    RealRawClaim p initial caller (rawRealCall p h (compiledTrials compiler p choices) (compiledChoice compiler p chosen) initial caller)

theorem bound_real : BoundReal :=
  ⟨real_dequant,
   fun implementation build p h a choices chosen initial caller =>
     real_s1 p h a (libraryPlan implementation p a build) choices chosen initial caller,
   fun p dtype _ h a order initial caller => real_quant p dtype h a order initial caller,
   fun compiler p h choices chosen initial caller =>
     real_raw p h (compiledTrials compiler p choices) (compiledChoice compiler p chosen) initial caller⟩

#axiomsClean bound_native
#axiomsClean bound_if_local
#axiomsClean bound_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.Direct
