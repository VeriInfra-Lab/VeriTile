import bench.tritonbench_g.mean_reduction.Proofs.WholeFPHost
import bench.tritonbench_g.mean_reduction.Proofs.WholeNative
import VeriTile.Meta.Specification

namespace VeriTile.Bench.TritonBenchG.MeanReduction.Whole
open VeriTile VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

inductive DimensionArgument where
  | single (dim : Int)
  | multiple (dims : List Int)

def DimensionArgument.list : DimensionArgument → List Int
  | .single d => [d]
  | .multiple ds => ds

/-- Python modulo is applied to each original axis after int-to-list conversion.
At rank zero only the empty list avoids Python's modulo-by-zero exception. -/
def normalizeArguments (rank : Nat) (arg : DimensionArgument) : Option (List (Fin rank)) :=
  if h : 0<rank then some (arg.list.map (normalizeAxis rank h))
  else if arg.list=[] then some [] else none

inductive DTypeArgument where
  | defaultFP32
  | explicitFP32

def DTypeArgument.output : DTypeArgument → Storage
  | .defaultFP32 | .explicitFP32 => .fp32

/-- Source argument admission only; it contains no runtime or computed values.
Repeated normalized axes are excluded because the source permute raises.
Empty explicit axis selection is the coordinatewise identity reduction. -/
structure SourceArguments (rank : Nat) (dims : List (Fin rank)) where
  raw : DimensionArgument
  normalized : normalizeArguments rank raw=some dims
  dtype : DTypeArgument

theorem normalize_single (rank : Nat) (h : 0<rank) (d : Int) :
    normalizeArguments rank (.single d)=some [normalizeAxis rank h d] := by
  simp [normalizeArguments,h,DimensionArgument.list]

theorem normalize_multiple (rank : Nat) (h : 0<rank) (ds : List Int) :
    normalizeArguments rank (.multiple ds)=some (ds.map (normalizeAxis rank h)) := by
  simp [normalizeArguments,h,DimensionArgument.list]

theorem normalize_scalar_empty : normalizeArguments 0 (.multiple [])=some [] := rfl

theorem dtype_selected (arg : DTypeArgument) : arg.output=.fp32 := by cases arg <;> rfl

variable {α : Type} {rank : Nat} {x : Tensor rank} {dims : List (Fin rank)}
  {nodup : dims.Nodup} {entry : Nat → α} {preexisting : Nat → Prop}
  {c : Contiguous α x dims nodup entry preexisting}

def SetupFailed (rt : Runtime α x dims nodup entry preexisting) : Prop :=
  rt.compress=none ∨ ∃ c, rt.compress=some c ∧
    (rt.allocate c=none ∨ ∃ a, rt.allocate c=some a ∧ rt.launchReady c a=Bool.false)

/-- Complete returned tensor contract, including physical exterior and entry
frames. The runtime supplies only library copies, allocation and scheduling. -/
def ReturnedSpec (formula : Fin (rows x.shape dims) → α)
    (rt : Runtime α x dims nodup entry preexisting) (keepdim : Bool) (out : Output α) : Prop :=
  ∃ c a, rt.compress=some c ∧ rt.allocate c=some a ∧ rt.launchReady c a=Bool.true ∧
    out.base=a.base ∧ out.shape=outputShape x.shape dims keepdim ∧
    out.dtype=.fp32 ∧ out.device=x.device ∧
    (∀ row : Fin (rows x.shape dims), out.memory (out.base+row.val)=formula row) ∧
    (∀ addr, (∀ i, i<rows x.shape dims → addr≠a.base+i) → out.memory addr=a.memory addr) ∧
    (∀ addr, preexisting addr → out.memory addr=entry addr)

def HostOutcomes (formula : Fin (rows x.shape dims) → α)
    (rt : Runtime α x dims nodup entry preexisting) (keepdim : Bool) (result : Option (Output α)) : Prop :=
  match result with
  | none => SetupFailed rt
  | some out => ReturnedSpec formula rt keepdim out

theorem real_outcomes {entry : Nat → ℝ} (rt : Runtime ℝ x dims nodup entry preexisting)
    (keepdim : Bool) :
    HostOutcomes (fun row => mathematicalMean x.shape dims (x.read entry)
      ((batchLinear x.shape dims).symm row)) rt keepdim (executeReal rt keepdim) := by
  cases hc : rt.compress with
  | none =>
    simpa [executeReal,hc,HostOutcomes,SetupFailed] using Or.inl hc
  | some c =>
    cases ha : rt.allocate c with
    | none =>
      simp only [bind,Option.bind,executeReal,hc,ha,Option.bind_some,Option.bind_none,HostOutcomes]
      change SetupFailed rt
      exact Or.inr ⟨c,hc,Or.inl ha⟩
    | some a =>
      cases ready : rt.launchReady c a with
      | «false» =>
        simp only [bind,Option.bind,executeReal,hc,ha,ready,Option.bind_some,Bool.false_eq_true,if_false,HostOutcomes]
        exact Or.inr ⟨c,hc,Or.inr ⟨a,ha,ready⟩⟩
      | «true» =>
        obtain ⟨final,ran,values,frame,protectedFrame⟩ :=
          allocated_real_result a (rt.order c a) (rt.schedule c a)
        simp only [bind,Option.bind,executeReal,hc,ha,ready,Option.bind_some,if_true,ran,HostOutcomes]
        exact ⟨c,a,hc,ha,ready,rfl,rfl,rfl,rfl,values,frame,protectedFrame⟩

theorem fp_outcomes [Inhabited α] (A : Algebra α)
    (rt : Runtime α x dims nodup entry preexisting) (keepdim : Bool)
    (positive : 0<columns x.shape dims) :
    HostOutcomes (fun row => fpFormula A x dims nodup entry positive row.val)
      rt keepdim (executeFP A Bool.false rt keepdim) := by
  cases hc : rt.compress with
  | none =>
    simp only [bind,Option.bind,executeFP,hc,Option.bind_none,HostOutcomes]
    exact Or.inl hc
  | some c =>
    cases ha : rt.allocate c with
    | none =>
      simp only [bind,Option.bind,executeFP,hc,ha,Option.bind_some,Option.bind_none,HostOutcomes]
      change SetupFailed rt
      exact Or.inr ⟨c,hc,Or.inl ha⟩
    | some a =>
      cases ready : rt.launchReady c a with
      | «false» =>
        simp only [bind,Option.bind,executeFP,hc,ha,ready,Option.bind_some,Bool.false_eq_true,if_false,HostOutcomes]
        exact Or.inr ⟨c,hc,Or.inr ⟨a,ha,ready⟩⟩
      | «true» =>
        obtain ⟨final,ran,values,frame,protectedFrame⟩ :=
          allocated_fp_result A a positive (rt.order c a) (rt.schedule c a)
        simp only [bind,Option.bind,executeFP,hc,ha,ready,Option.bind_some,if_true,ran,HostOutcomes]
        exact ⟨c,a,hc,ha,ready,rfl,rfl,rfl,rfl,values,frame,protectedFrame⟩

theorem native_pointers (a : Allocated c) (f : FiniteAdmission a)
    (pid t lane col : Nat) (hp : pid<gridSize (rows x.shape dims))
    (ht : t<gridSize (columns x.shape dims)) (hl : lane<8) (hc : col<8) :
    (nativeInputPointer c.base pid lane (columns x.shape dims) (t*8) col).toNat =
      4*(c.base+(pid*8+lane)*columns x.shape dims+(t*8+col)) ∧
    (nativeOutputPointer a.base pid lane).toNat=4*(a.base+pid*8+lane) := by
  have ints := source_integer_correspondence a f pid t lane col hp ht hl hc
  have ptrs := source_pointer_correspondence a f pid t lane col hp ht hl hc
  simp only [nativeInputPointer,nativeOutputPointer,ints.1,ints.2.1,ints.2.2.1,Int.toNat_natCast]
  constructor
  · have eq : BitVec.ofNat 64 (4*c.base)+
      BitVec.ofNat 64 (4*((pid*8+lane)*columns x.shape dims))+
      BitVec.ofNat 64 (4*(t*8+col)) =
      bytePointer c.base ((pid*8+lane)*columns x.shape dims+(t*8+col)) := by
        unfold bytePointer
        rw [BitVec.ofNat_add_ofNat,BitVec.ofNat_add_ofNat,BitVec.ofNat_add_ofNat]
        congr 1
        omega
    rw [eq]
    exact ptrs.1
  · simpa only [bytePointer,Nat.add_assoc] using ptrs.2

/-- Consolidated finite-index, native pointer, original-input and active-cell
correspondence. Numerical payload behavior is absent from these obligations. -/
def IndexCorrespondence (a : Allocated c) (f : FiniteAdmission a) : Prop :=
  (∀ pid t lane col, pid<gridSize (rows x.shape dims) → t<gridSize (columns x.shape dims) →
    lane<8 → col<8 →
    (nativeInputPointer c.base pid lane (columns x.shape dims) (t*8) col).toNat =
      4*(c.base+(pid*8+lane)*columns x.shape dims+(t*8+col)) ∧
    (nativeOutputPointer a.base pid lane).toNat=4*(a.base+pid*8+lane) ∧
    (rowBits pid lane).slt (i32 (rows x.shape dims))=decide (pid*8+lane<rows x.shape dims) ∧
    (colBits (t*8) col).slt (i32 (columns x.shape dims))=decide (t*8+col<columns x.shape dims)) ∧
  (∀ i, 4*(x.address i)+4 ≤ 2^64 ∧ preexisting (x.address i)) ∧
  (∀ row col, row<rows x.shape dims → col<columns x.shape dims →
    row*columns x.shape dims+col<rows x.shape dims*columns x.shape dims ∧
    4*(c.base+row*columns x.shape dims+col)+4 ≤ 2^64 ∧ 4*(a.base+row)+4 ≤ 2^64) ∧
  (∀ row addr i j, row<rows x.shape dims → preexisting addr → i<4 → j<4 →
    4*(a.base+row)+i≠4*addr+j)

theorem index_correspondence (a : Allocated c) (f : FiniteAdmission a) : IndexCorrespondence a f := by
  refine ⟨?_,original_accesses a f,?_,output_preexisting_bytes_separate a⟩
  · intro pid t lane col hp ht hl hc
    have ptrs := native_pointers a f pid t lane col hp ht hl hc
    have ints := source_integer_correspondence a f pid t lane col hp ht hl hc
    refine ⟨ptrs.1,ptrs.2,?_,?_⟩
    · exact Bool.eq_iff_iff.mpr (by simpa using ints.2.2.2.1)
    · exact Bool.eq_iff_iff.mpr (by simpa using ints.2.2.2.2.1)
  · intro row col hr hc
    have h := active_accesses a f row col hr hc
    exact ⟨h.1,h.2.1,h.2.2.1⟩

/-- Additive public mathematical claim: original complete host/source/grid,
independent logical-axis mean, every return/failure, full frame and finite addresses. -/
specification mean_dim_real_source_outcomes {entry : Nat → ℝ}
    (_arguments : SourceArguments rank dims) (positive : 0<columns x.shape dims)
    (rt : Runtime ℝ x dims nodup entry preexisting) (keepdim : Bool)
    (finite : ∀ c a, rt.compress=some c → rt.allocate c=some a →
      rt.launchReady c a=Bool.true → FiniteAdmission a) :
    Spec.Real (HostOutcomes (fun row => mathematicalMean x.shape dims (x.read entry)
      ((batchLinear x.shape dims).symm row)) rt keepdim (executeReal rt keepdim) ∧
      ∀ c a hc ha ready, IndexCorrespondence a (finite c a hc ha ready)) := by
  exact ⟨real_outcomes rt keepdim,fun c a hc ha ready => index_correspondence a (finite c a hc ha ready)⟩

/-- Supporting generic source interpretation on every BitVec32 payload.
The primary native specification in WholeNativeHost instantiates explicit
observed graphs, with only three scalar instruction families left open. -/
theorem mean_dim_fp32_source_outcomes {entry : Nat → BitVec 32} (A : Algebra (BitVec 32))
    (_arguments : SourceArguments rank dims) (positive : 0<columns x.shape dims)
    (rt : Runtime (BitVec 32) x dims nodup entry preexisting) (keepdim : Bool)
    (finite : ∀ c a, rt.compress=some c → rt.allocate c=some a →
      rt.launchReady c a=Bool.true → FiniteAdmission a) :
    HostOutcomes (fun row => fpFormula A x dims nodup entry positive row.val)
      rt keepdim (executeFP A Bool.false rt keepdim) ∧
    executeFP A Bool.false rt keepdim=executeFP A Bool.true rt keepdim ∧
    ∀ c a hc ha ready, IndexCorrespondence a (finite c a hc ha ready) := by
  exact ⟨fp_outcomes A rt keepdim positive,executeFP_equal A rt keepdim,
    fun c a hc ha ready => index_correspondence a (finite c a hc ha ready)⟩

#axiomsClean mean_dim_real_source_outcomes
#axiomsClean mean_dim_fp32_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MeanReduction.Whole
