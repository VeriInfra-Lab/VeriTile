import bench.repairs.tritonbench_g.bmm_chunk_bwd.Source
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness

/-! FP whole-grid composition. Numerical guards remain at every actual reached
residual addition in the evolving original execution. This preserves the
complete typed state and failure behavior; it introduces no dot reassociation.
-/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
noncomputable section

/-- Same host arguments as the real entry, retaining storage and all casts. -/
def original (h : HostCall) (c : Tuning) : Program where
  kernel := h.source c Bool.false
  domain := FPEquivProofs.requirementsFor (h.source c Bool.false) c.BM c.BN

def residualLeft (h : HostCall) (c : Tuning) : Program :=
  { original h c with kernel := h.source c Bool.true }

theorem program_equiv (h : HostCall) (c : Tuning) (R : FP.ScalarArithmetic.Rules) :
    original h c ≡[R] residualLeft h c := by
  exact FPEquivProofs.residual_equiv h.aStorage h.doutStorage h.residualDType h.outputDType
    h.A h.Dout h.Out h.Residual h.seqlen h.chunk h.K h.groupCount
    h.aStrides.batch h.aStrides.sequence (h.headStride h.aStrides.head) h.aStrides.feature
    h.doutStrides.batch h.doutStrides.chunk (h.headStride h.doutStrides.head) h.doutStrides.row h.doutStrides.column
    h.outStrides.batch h.outStrides.sequence (h.headStride h.outStrides.head) h.outStrides.feature
    (h.resStride h.residualStrides.batch) (h.resStride h.residualStrides.sequence)
    (h.resStride (h.headStride h.residualStrides.head)) (h.resStride h.residualStrides.feature)
    c.BM c.BN c.BCS h.hasResidual R

abbrev Pid := Nat × Nat × Nat

def grid (h : HostCall) (c : Tuning) : List Pid :=
  (List.range (((h.chunk + c.BM - 1) / c.BM) * ((h.K + c.BN - 1) / c.BN))).product
    ((List.range h.batch).product (List.range (h.nchunks * h.groupCount)))

def launch {α : Type} (h : HostCall) (c : Tuning) (q : Pid) (s : State α) : State α :=
  { s with regs := fun _ _ _ => none
           pids := fun ax => if ax = 0 then q.1 else if ax = 1 then q.2.1 else if ax = 2 then q.2.2 else 0
           numPids := fun ax =>
             if ax = 0 then ((h.chunk + c.BM - 1) / c.BM) * ((h.K + c.BN - 1) / c.BN)
             else if ax = 1 then h.batch else if ax = 2 then h.nchunks * h.groupCount else 1 }

def run {α : Type} [Inhabited α] (M : Algebra α) (h : HostCall) (c : Tuning) (swapped : Bool) :
    List Pid → State α → Option (State α)
  | [], s => some s
  | q :: rest, s => do
      let next ← FP.Structural.exec (engine M) (h.source c swapped) (launch h c q s)
      run M h c swapped rest next

/-- Guards only on states actually reached by the original prefix. This does
not require or assume successful execution or an output equality. -/
def ReachedDomain {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α)
    (h : HostCall) (c : Tuning) : List Pid → State α → Prop
  | [], _ => True
  | q :: rest, s =>
      (∀ site ∈ (original h c).domain, site.Holds M D (launch h c q s)) ∧
      ∀ next, FP.Structural.exec (engine M) (h.source c Bool.false) (launch h c q s) = some next →
        ReachedDomain M D h c rest next

theorem run_equiv (R : FP.ScalarArithmetic.Rules) {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α) (model : FP.Guarded.Models R M D)
    (h : HostCall) (c : Tuning) (jobs : List Pid) (s : State α)
    (domain : ReachedDomain M D h c jobs s) :
    run M h c Bool.false jobs s = run M h c Bool.true jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons q rest ih =>
    have eq := sound (program_equiv h c R) α M D model (launch h c q s) domain.1
    change FP.Structural.exec (engine M) (h.source c Bool.false) (launch h c q s) =
      FP.Structural.exec (engine M) (h.source c Bool.true) (launch h c q s) at eq
    simp only [run]
    rw [← eq]
    cases he : FP.Structural.exec (engine M) (h.source c Bool.false) (launch h c q s) with
    | none => rfl
    | some next => exact ih next (domain.2 next he)

#print_fp_assumptions program_equiv
#axiomsClean run_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.HostFP
