import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Equivalence
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.ScalarReal

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelScalar
open VeriTile.Triton FP.Structural Packing Storage
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 64000
set_option linter.unusedSimpArgs false

/-- Unreached operations have a fixed interpretation. The imported scalar
proofs show the selected source outputs use only the concrete covered fragment. -/
def fallback (zero : α) : Algebra α where
  literal:=fun _ _ _=>zero
  negInf:=zero
  binary:=fun _ _ _ _ _=>zero
  unary:=fun _ _ x=>x
  cast:=fun _ _ _ x=>x
  fromNat:=fun _ _=>zero
  fromInt:=fun _ _=>zero
  fp32Bits:=fun _=>zero
  fp32Load:=id
  reduceMax:=fun _ _ _ _ _ _=>zero
  reduceSum:=fun _ _ _ _ _ _=>zero

def realBackend : Algebra ℝ := ScalarReal.backend (fallback 0)
def wordBase : Algebra Packing.Word := fallback (fun _=>0)
def valueBase : Algebra (FP.Value .fp32) := fallback (FP.Value.zero .fp32)

/-- Both actual Python launches explicitly disable FP fusion. The concrete
selected model rounds each FP32 add/sub/mul separately, nearest-even without
FTZ. This model declaration is not GPU/compiler implementation conformance. -/
def selectedProfile : FP.Config := .ieeeValue
def forwardBackend (cfg : FP.Config) : Algebra Packing.Word :=
  FPProof.transport wordBase (Software.forwardBackend valueBase cfg)
def backwardBackend (cfg : FP.Config) : Algebra Packing.Word :=
  FPProof.transport wordBase (Software.backend valueBase cfg)

theorem local_self (M : Algebra α) (inputs : Nat → α) (e : Arithmetic.Expr) :
    Arithmetic.LocalNumerics M M inputs e := by
  induction e with
  | input _=>trivial
  | literal _=>rfl
  | binary _ _ _ a b=>exact ⟨a,b,rfl⟩
  | unary _ _ a=>exact ⟨a,rfl⟩

/-- Historical primitive-local obligations are discharged here by selecting
their concrete implementation. They are not premises of the new public entries. -/
theorem forward_laws (cfg : FP.Config) (inputs : Nat → Packing.Word) (c : Fin 11) :
    FPProof.ForwardNumerics (Equivalence.exactLoads (forwardBackend cfg)) wordBase valueBase cfg inputs c :=
  local_self (forwardBackend cfg) inputs (Arithmetic.forwardGraph c)
theorem backward_laws (cfg : FP.Config) (inputs : Nat → Packing.Word) (c : Fin 3) :
    FPProof.BackwardNumerics (Equivalence.exactLoads (backwardBackend cfg)) wordBase valueBase cfg inputs c :=
  local_self (backwardBackend cfg) inputs (Arithmetic.backwardGraph c)

def forwardReference (cfg : FP.Config) (b : Physical.Buffers) (heap : Heap) (r : Nat) : Fin 11 → FP.Value .fp32 :=
  Software.forward cfg
    (Packing.value (Equivalence.forwardInputs b heap r 0))
    (Packing.value (Equivalence.forwardInputs b heap r 1))
    (Packing.value (Equivalence.forwardInputs b heap r 2))
def backwardReference (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap) (r : Nat) : Fin 3 → FP.Value .fp32 :=
  Software.backward cfg
    (Packing.value (Equivalence.backwardInputs p b heap r 0))
    (Packing.value (Equivalence.backwardInputs p b heap r 1))
    (Packing.value (Equivalence.backwardInputs p b heap r 2))
    (fun c=>Packing.value (Equivalence.backwardInputs p b heap r (3+c.val)))
    (fun c=>Packing.value (Equivalence.backwardInputs p b heap r (14+c.val)))

def Frame (d : Grid.Direction) (p : Parameters) (before final : State α) : Prop :=
  ∀ region index,(region≠Grid.output d ∨ ∀ r,r<p.rows → ∀ c : Fin (Grid.channels d),
    index≠Grid.offset p d r c.val) → final.mem region index=before.mem region index

theorem empty [Inhabited α] (M : Algebra α) (d : Grid.Direction) (p : Parameters)
    (before : State α) (rows : p.rows=0) : Grid.launch M d p before=some before := by
  have positive:=p.blockPositive
  have blocks : Grid.blocks p=0 := by
    unfold Grid.blocks
    rw [rows]
    exact Nat.div_eq_of_lt (by omega)
  simp only [Grid.launch,blocks,List.range_zero,Grid.execute]

theorem native_forward (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ final,Grid.launch (forwardBackend cfg) .forward p (Physical.backed .forward p b heap)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 11,final.mem outputRegion (r*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (Packing.word (forwardReference cfg b heap r c))) ∧
      Frame .forward p (Physical.backed .forward p b heap) final := by
  obtain ⟨final,run,values,frame⟩:=Grid.launch_correct (forwardBackend cfg) .forward p
    (Physical.backed .forward p b heap) fit
  refine ⟨final,run,?_,frame⟩
  intro r hr c
  exact (values r hr c).trans (congrArg (Cell.mk .real)
    (Equivalence.forward_value (forwardBackend cfg) wordBase valueBase cfg p b heap r c
      (forward_laws cfg _ c)))

theorem native_backward (cfg : FP.Config) (p : Parameters) (b : Physical.Buffers) (heap : Heap)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ final,Grid.launch (backwardBackend cfg) .backward p (Physical.backed .backward p b heap)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 3,final.mem gradientRegion (r*3+c.val)=
        .mk .real (Packing.word (backwardReference cfg p b heap r c))) ∧
      Frame .backward p (Physical.backed .backward p b heap) final := by
  obtain ⟨final,run,values,frame⟩:=Grid.launch_correct (backwardBackend cfg) .backward p
    (Physical.backed .backward p b heap) fit
  refine ⟨final,run,?_,frame⟩
  intro r hr c
  exact (values r hr c).trans (congrArg (Cell.mk .real)
    (Equivalence.backward_value (backwardBackend cfg) wordBase valueBase cfg p b heap r c
      (backward_laws cfg _ c)))

/-- Ideal Real state reads the actual prepared destination too. Only the
active initial gradient words will be required to be zero; padding is arbitrary. -/
def realInitial (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ) : State ℝ where
  mem region index:=.mk .real (memory ((if region=coordinateRegion then b.coordinates
    else if region=upstreamRegion then b.upstream else b.destination)+4*index))
  regs _ _ _:=none
  pids _:=0
  numPids dim:=if dim=0 then Grid.blocks p else 1
  undef dtype _ _:=defaultValue dtype

def realForward (b : Physical.Buffers) (memory : Nat → ℝ) (r : Nat) : Fin 11 → ℝ :=
  fun c=>Polynomial.value c (memory (b.coordinates+4*(r*3)))
    (memory (b.coordinates+4*(r*3+1))) (memory (b.coordinates+4*(r*3+2)))
def realBackward (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ) (r : Nat) : Fin 3 → ℝ :=
  Polynomial.vjp (memory (b.coordinates+4*(r*3)))
    (memory (b.coordinates+4*(r*3+1))) (memory (b.coordinates+4*(r*3+2)))
    (fun c=>memory (b.upstream+4*(r*p.upstreamRowStride+(p.columnOffset+c.val)*p.upstreamColStride)))

theorem real_forward_value (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ)
    (r : Nat) (c : Fin 11) :
    Grid.value realBackend .forward p (realInitial p b memory) r c=realForward b memory r c := by
  simp only [Grid.value,realInitial,realBackend,ScalarReal.backend,realForward,
    coordinateRegion,Cell.read_same,Fin.val_zero,Fin.val_one,Fin.val_two,Nat.add_zero,ite_true]
  exact ScalarReal.forward (fallback 0) _ _ _ c

theorem real_backward_value (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ)
    (r : Nat) (zero : ∀ c : Fin 3,memory (b.destination+4*(r*3+c.val))=0) (c : Fin 3) :
    Grid.value realBackend .backward p (realInitial p b memory) r c=realBackward p b memory r c := by
  change Reference.backwardValues realBackend
    (memory (b.coordinates+4*(r*3)))
    (memory (b.coordinates+4*(r*3+1)))
    (memory (b.coordinates+4*(r*3+2)))
    (fun j=>memory (b.upstream+4*(r*p.upstreamRowStride+(p.columnOffset+j.val)*p.upstreamColStride)))
    (fun j=>memory (b.destination+4*(r*3+j.val))) c=realBackward p b memory r c
  have prepared : (fun j : Fin 3=>memory (b.destination+4*(r*3+j.val)))=(fun _=>0) := funext zero
  rw [prepared]
  exact ScalarReal.backward (fallback 0) _ _ _ _ c

theorem real_forward (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ)
    (fit : p.columnOffset+11≤p.outputWidth) :
    ∃ final,Grid.launch realBackend .forward p (realInitial p b memory)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 11,final.mem outputRegion (r*p.outputWidth+p.columnOffset+c.val)=
        .mk .real (realForward b memory r c)) ∧ Frame .forward p (realInitial p b memory) final := by
  obtain ⟨final,run,values,frame⟩:=Grid.launch_correct realBackend .forward p (realInitial p b memory) fit
  exact ⟨final,run,fun r hr c=>(values r hr c).trans
    (congrArg (Cell.mk .real) (real_forward_value p b memory r c)),frame⟩

theorem real_backward (p : Parameters) (b : Physical.Buffers) (memory : Nat → ℝ)
    (fit : p.columnOffset+11≤p.outputWidth)
    (zero : ∀ r,r<p.rows → ∀ c : Fin 3,memory (b.destination+4*(r*3+c.val))=0) :
    ∃ final,Grid.launch realBackend .backward p (realInitial p b memory)=some final ∧
      (∀ r,r<p.rows → ∀ c : Fin 3,final.mem gradientRegion (r*3+c.val)=
        .mk .real (realBackward p b memory r c)) ∧ Frame .backward p (realInitial p b memory) final := by
  obtain ⟨final,run,values,frame⟩:=Grid.launch_correct realBackend .backward p (realInitial p b memory) fit
  exact ⟨final,run,fun r hr c=>(values r hr c).trans
    (congrArg (Cell.mk .real) (real_backward_value p b memory r (zero r hr) c)),frame⟩

#axiomsClean native_forward
#axiomsClean native_backward
#axiomsClean real_forward
#axiomsClean real_backward
#axiomsClean empty
#specNonCircular forwardReference avoiding [forwardKernel,backwardKernel,Grid.launch]
#specNonCircular backwardReference avoiding [forwardKernel,backwardKernel,Grid.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.KernelScalar
