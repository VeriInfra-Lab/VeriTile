import bench.repairs.tritonbench_g.index_select_cat.Proofs.RawKernel
import bench.repairs.tritonbench_g.index_select_cat.Contract
import bench.repairs.tritonbench_g.index_select_cat.Proofs.TypedGather

namespace VeriTile.Bench.Repairs.IndexSelectCat.KernelProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]
abbrev Lane := TileIndex [1,512]

theorem checks_iff (r : Request) : checkInputs r = Bool.true ↔ InputContract r := by
  simp only [checkInputs,Bool.and_eq_true,decide_eq_true_eq,InputContract,and_assoc]

theorem argument_checks_iff (a : Arguments) :
    checkArguments a = Bool.true ↔ AdmittedArguments a := by
  cases a with
  | tensors r => exact checks_iff r
  | invalid => simp [checkArguments,AdmittedArguments]

def expected (r : Request) (s : State α) (i d : Nat) : α :=
  target r.indices (fun row col => (s.mem r.layout.source (row*r.D+col)).read .real) i d

def Written (r : Request) (pid : Nat → Nat) (off : Nat) : Prop :=
  ∃ d : Fin 512,pid 0 < r.N ∧ pid 1*512+d.val < r.D ∧
    off = pid 0*r.D+(pid 1*512+d.val)

theorem lane_offsets (r : Request) (s : State α) (i : Lane) :
    TypedGather.indexOffset r.layout s i = s.pids 0 ∧
    TypedGather.column r.layout s i = s.pids 1*512+i.2.1.val ∧
    TypedGather.outputOffset r.layout s i = s.pids 0*r.D+(s.pids 1*512+i.2.1.val) ∧
    (TypedGather.active r.layout s i ↔ s.pids 0 < r.N ∧ s.pids 1*512+i.2.1.val < r.D) := by
  have iz : i.1.val = 0 := by have := i.1.isLt; omega
  simp [TypedGather.indexOffset,TypedGather.column,TypedGather.outputOffset,
    TypedGather.active,Request.layout,iz]

theorem index_valid (r : Request) (h : InputContract r) (pid : Nat → Nat)
    (launch : Launched r pid) :
    0 ≤ r.indices (pid 0) ∧ (r.indices (pid 0)).toNat < r.M := by
  have positive : 0 < r.D := by rcases launch with ⟨_,hc⟩; omega
  have valid := h.2.2.resolve_left (by omega) ⟨pid 0,launch.1⟩
  change 0 ≤ r.indices (pid 0) ∧ r.indices (pid 0) < (r.M : Int) at valid
  exact ⟨valid.1,by omega⟩

theorem input_offset (r : Request) (h : InputContract r) (s : State α)
    (launch : Launched r s.pids)
    (observed : IndexObservation r (fun i => (s.mem r.layout.index i).read .int)) (i : Lane) :
    TypedGather.inputOffset r.layout s i =
      (r.indices (s.pids 0)).toNat*r.D+(s.pids 1*512+i.2.1.val) := by
  have offsets := lane_offsets r s i
  have valid := index_valid r h s.pids launch
  have row : TypedGather.row r.layout s i = r.indices (s.pids 0) := by
    simp only [TypedGather.row,offsets.1]
    exact observed _ launch.1
  unfold TypedGather.inputOffset
  rw [row,offsets.2.1]
  simp only [Request.layout,Nat.mul_one]
  rw [Int.toNat_add (Int.mul_nonneg valid.1 (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul valid.1 (Int.natCast_nonneg _)]
  simp only [Int.toNat_natCast]

/-- Every actual masked source access is inside the checked initial Tensor
window. Index bounds are independent of column activity because the index load
precedes the column mask. The metadata observation binds the actual ID load. -/
theorem accesses (r : Request) (h : InputContract r) (s : State α)
    (launch : Launched r s.pids)
    (observed : IndexObservation r (fun i => (s.mem r.layout.index i).read .int)) :
    (∀ i : Lane,TypedGather.indexOffset r.layout s i < r.N) ∧
    (∀ i : Lane,TypedGather.active r.layout s i →
      TypedGather.inputOffset r.layout s i < r.M*r.D ∧
      TypedGather.outputOffset r.layout s i < r.N*r.D) := by
  refine ⟨fun i => by rw [(lane_offsets r s i).1]; exact launch.1,?_⟩
  intro i active
  have offsets := lane_offsets r s i
  have col := (offsets.2.2.2.mp active).2
  have valid := index_valid r h s.pids launch
  have rowBound := launch.1
  rw [input_offset r h s launch observed i,offsets.2.2.1]
  constructor <;> nlinarith [valid.2]

theorem output_unique (r : Request) (s : State α) (a b : Lane)
    (eq : TypedGather.outputOffset r.layout s a = TypedGather.outputOffset r.layout s b) : a = b := by
  rw [(lane_offsets r s a).2.2.1,(lane_offsets r s b).2.2.1] at eq
  apply Prod.ext (Subsingleton.elim _ _)
  apply Prod.ext _ (Subsingleton.elim _ _)
  apply Fin.ext
  omega

/-- Constructive success, initial-input gather formula, and exterior frame for
the exact selected source, starting with arbitrary registers and output cells. -/
theorem correct (rest : Algebra α) (r : Request) (h : InputContract r) (s : State α)
    (launch : Launched r s.pids)
    (observed : IndexObservation r (fun i => (s.mem r.layout.index i).read .int)) :
    ∃ t,FP.Structural.exec (copyBackend rest) (kernel r.layout) s = some t ∧
      (∀ d : Fin 512,s.pids 1*512+d.val < r.D →
        t.mem r.layout.output (s.pids 0*r.D+(s.pids 1*512+d.val)) =
          .mk .real (expected r s (s.pids 0) (s.pids 1*512+d.val))) ∧
      (∀ region off,region ≠ r.layout.output ∨ ¬Written r s.pids off →
        t.mem region off = s.mem region off) := by
  obtain ⟨t,run,output,frame⟩ := RawKernel.complete rest r.layout s
    (fun a b _ _ eq => output_unique r s a b eq)
  refine ⟨t,run,?_,?_⟩
  · intro d hd
    let i : Lane := (⟨0,by omega⟩,d,())
    have active : TypedGather.active r.layout s i := (lane_offsets r s i).2.2.2.mpr ⟨launch.1,hd⟩
    have readback : t.mem r.layout.output (TypedGather.outputOffset r.layout s i) =
        TypedGather.loaded r.layout s i := by
      exact output i active
    rw [(lane_offsets r s i).2.2.1] at readback
    simpa only [TypedGather.loaded,input_offset r h s launch observed i,expected,target,i] using readback
  · intro region off outside
    apply frame
    rcases outside with outside | outside
    · exact Or.inl outside
    · right
      intro i active eq
      have offsets := lane_offsets r s i
      exact outside ⟨i.2.1,(offsets.2.2.2.mp active).1,(offsets.2.2.2.mp active).2,
        eq.trans offsets.2.2.1⟩

#specNonCircular expected avoiding [kernel,indexCommutedKernel,FP.Structural.exec]
#axiomsClean accesses
#axiomsClean correct
end
end VeriTile.Bench.Repairs.IndexSelectCat.KernelProof
