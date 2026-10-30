import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Native
import bench.repairs.tritonbench_g.swiglu_triton.Proofs.Source

namespace VeriTile.Bench.Repairs.SwigluTriton.Native
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Observed (Term graphAlgebra)
open Memory Physical
noncomputable section

abbrev Word := BitVec 32

def positiveZero : Word := BitVec.ofNat 32 0x00000000
def one : Word := BitVec.ofNat 32 0x3f800000
def log2e : Word := BitVec.ofNat 32 0x3fb8aa3b

/-- Precision/configuration key of the actual compiler-resolved instruction
profile. The 14 evidence samples instantiate CUDA/sm90/Triton3.7.1 and the
unchanged source options; other profiles are not inferred from those samples. -/
structure Profile where
  backend : String
  architecture : Nat
  compiler : String
  fpFusion : Bool
  flushToZero : Bool
  columns : Nat
  block : Nat
  warps : Nat
  stride : Nat

/-- Independently specified scalar instruction semantics. Their exact
native-to-target correspondence is the only numerical boundary:
round-to-nearest FP32 add/sub/mul/FMA, PTX div.full.f32, ex2.approx.f32.
All ports take raw32 words, retaining nonfinite values, NaNs and signed zeros. -/
structure InstructionSemantics where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  fmaRN : Word → Word → Word → Word
  divFull : Word → Word → Word
  exp2Approx : Word → Word

/-- Exact sigmoid subgraph observed in every pinned PTX profile. Negation is
sub(+0,a), not an assumed real negation, and log2(e) is the encoded constant. -/
def sigmoidGraph (p : InstructionSemantics) (a : Word) : Word :=
  p.divFull one (p.addRN (p.exp2Approx (p.mulRN (p.subRN positiveZero a) log2e)) one)

def compiledPorts (p : InstructionSemantics) : Primitives Word where
  literal := fun x => if x=1 then one else positiveZero
  cast := id
  sigmoid := sigmoidGraph p
  sub := p.subRN
  mul := p.mulRN
  fma := p.fmaRN
  unused := positiveZero

/-- This independent FP reference explicitly fixes the compiled fused
policy and all operand orders; it is not source split arithmetic. -/
def compiledReference (p : InstructionSemantics) (ch : Channel) (a b dc : Word) : Word :=
  let s := p.divFull one (p.addRN (p.exp2Approx (p.mulRN (p.subRN positiveZero a) log2e)) one)
  let silu := p.mulRN s a
  match ch with
  | .c => p.mulRN silu b
  | .da => p.mulRN (p.mulRN (p.fmaRN silu (p.subRN one s) s) dc) b
  | .db => p.mulRN silu dc

theorem compiled_lowering (p : InstructionSemantics) (ch : Channel) (a b dc : Word) :
    sourceScalar (compiledPorts p) ch a b dc=compiledReference p ch a b dc := by
  rw [graph_lowering]
  cases ch <;> simp [reference,compiledReference,compiledPorts,sigmoidGraph]

/-- Exact local obligations for one resolved profile. No whole sigmoid,
SwiGLU scalar, memory operation, launch or host result is an assumption. -/
structure PrimitiveObligations (profile : Profile) (native target : InstructionSemantics) : Prop where
  addRN : ∀ a b,native.addRN a b=target.addRN a b
  subRN : ∀ a b,native.subRN a b=target.subRN a b
  mulRN : ∀ a b,native.mulRN a b=target.mulRN a b
  fmaRN : ∀ a b c,native.fmaRN a b c=target.fmaRN a b c
  divFull : ∀ a b,native.divFull a b=target.divFull a b
  exp2Approx : ∀ a,native.exp2Approx a=target.exp2Approx a

theorem primitive_bridge (profile : Profile) (native target : InstructionSemantics)
    (h : PrimitiveObligations profile native target) (ch : Channel) (a b dc : Word) :
    sourceScalar (compiledPorts native) ch a b dc=compiledReference target ch a b dc := by
  rw [compiled_lowering]
  cases ch <;> simp only [compiledReference,h.addRN,h.subRN,h.mulRN,h.fmaRN,h.divFull,h.exp2Approx]

/-- Same actual caller/private word reads become atoms in the original
full typed source execution. Only node interpretation supplies instructions. -/
def graphState {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) : State (Term Word) where
  mem := fun reg offset => .mk .real (.input (
    if reg="A" then ops.read s ((p.allocation .a).key (first*p.block+offset))
    else if reg="B" then ops.read s ((p.allocation .b).key (first*p.block+offset))
    else if reg=Source.region backward .third then ops.read s ((p.allocation .third).key (first*p.block+offset))
    else positiveZero))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun dtype _ _ => defaultValue dtype

theorem source_row_native {M β : Type} (ops : Ops M Word β) (instructions : InstructionSemantics)
    (backward : Bool) (p : Layout) (valid : Valid p) (first pid : Nat)
    (rowBound : first+pid<p.rows) (s : M) (ch : Channel) (on : enabled backward ch)
    (col : Nat) (active : col<p.cols) :
    let execution := Tile.sourceRun graphAlgebra backward p.block p.cols pid (graphState ops backward p first s)
    interpret (compiledPorts instructions) ((execution.val.mem (outputRegion ch) (pid*p.block+col)).read .real)=
      ops.read (Physical.row ops (sourceScalar (compiledPorts instructions)) backward p (first+pid) s)
        (p.key (outputRole ch) (first+pid) col) := by
  dsimp only
  rw [Tile.readback graphAlgebra backward p.block p.cols pid (graphState ops backward p first s) ch on
    (⟨⟨col,lt_of_lt_of_le active valid.cols⟩,PUnit.unit⟩) active]
  rw [Physical.row_values ops _ backward p valid (first+pid) s rowBound ch on col active]
  cases backward <;> cases ch <;> try simp [enabled] at on
  all_goals simp [graphState,Source.region,Physical.value,Layout.key,Nat.add_mul,Nat.add_assoc,
    sourceScalar,Tile.scalar,graphAlgebra,interpret,compiledPorts,sigmoidGraph]

theorem graphState_read {M β : Type} (ops : Ops M Word β) (backward : Bool)
    (p : Layout) (first : Nat) (s : M) (role : Role) (offset : Nat) :
    (graphState ops backward p first s).mem (Source.region backward role) offset=
      Cell.mk .real (.input (ops.read s ((p.allocation role).key (first*p.block+offset)))) := by
  cases backward <;> cases role <;> simp [graphState,Source.region]

theorem source_row_native_full {M β : Type} (ops : Ops M Word β) (instructions : InstructionSemantics)
    (backward : Bool) (p : Layout) (valid : Valid p) (first pid : Nat)
    (rowBound : first+pid<p.rows) (s : M) :
    ∃ t,run graphAlgebra (source backward p.block p.cols).surfaceBody
        (Tile.rowState pid (graphState ops backward p first s))=some t ∧
      ∀ role offset,first*p.block+offset<(p.allocation role).words →
        interpret (compiledPorts instructions) ((t.mem (Source.region backward role) offset).read .real)=
          ops.read (Physical.row ops (sourceScalar (compiledPorts instructions)) backward p (first+pid) s)
            ((p.allocation role).key (first*p.block+offset)) := by
  refine ⟨_,(Tile.sourceRun graphAlgebra backward p.block p.cols pid (graphState ops backward p first s)).property,?_⟩
  intro role offset backed
  by_cases hit : ∃ ch,enabled backward ch ∧ role=outputRole ch ∧
      ∃ col,col<p.cols ∧ offset=pid*p.block+col
  · obtain ⟨ch,on,same,col,active,equal⟩ := hit
    subst role offset
    rw [Source.output_region backward ch on]
    have keyEq : ((p.allocation (outputRole ch)).key (first*p.block+(pid*p.block+col)))=
        p.key (outputRole ch) (first+pid) col := by simp [Layout.key,Nat.add_mul,Nat.add_assoc]
    rw [keyEq]
    exact source_row_native ops instructions backward p valid first pid rowBound s ch on col active
  · have structural : Tile.Outside backward p.block p.cols pid (Source.region backward role) offset := by
      intro ch on
      by_cases different : Source.region backward role≠outputRegion ch
      · exact Or.inl different
      · right
        intro lane active equal
        have sameRole : role=outputRole ch := Source.region_injective backward
          ((not_not.mp different).trans (Source.output_region backward ch on).symm)
        exact hit ⟨ch,on,sameRole,lane.1.val,active,equal⟩
    rw [Tile.frame graphAlgebra backward p.block p.cols pid (graphState ops backward p first s) _ _ structural,
      graphState_read]
    change ops.read s _=ops.read _ _
    symm
    apply Physical.row_frame
    intro ch on col active
    by_cases sameRole : role=outputRole ch
    · subst role
      apply Allocation.indices_disjoint
      intro words
      have sameOffset : offset=pid*p.block+col := by simp only [Nat.add_mul] at words; omega
      exact hit ⟨ch,on,rfl,col,active,sameOffset⟩
    · exact Allocation.spans_separate _ _ (valid.separate role (outputRole ch) sameRole) _ _
        (Allocation.key_contains _ _ backed)
        (key_backed p valid (outputRole ch) (first+pid) col rowBound (lt_of_lt_of_le active valid.cols))

#specNonCircular compiledReference avoiding [interpret,sourceScalar,Tile.scalar,Tile.sourceRun]
#axiomsClean primitive_bridge
#axiomsClean source_row_native_full
#axiomsClean source_row_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton.Native
