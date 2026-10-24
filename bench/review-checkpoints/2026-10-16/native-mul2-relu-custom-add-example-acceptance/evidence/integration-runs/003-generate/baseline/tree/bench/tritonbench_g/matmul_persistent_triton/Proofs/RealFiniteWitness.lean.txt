import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFull
import VeriTile.Triton.Memory.Denotation

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealFiniteWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000

/-- Two persistent output tiles and two K blocks with tails in M/N/K, plus
noncontiguous strides. All untouched cells initially carry the int tag. -/
def sample : Parameters where
  A := "A"; B := "B"; C := "C"
  M := 3; N := 3; K := 3
  sam := 9; sak := 2; sbk := 11; sbn := 2; scm := 7; scn := 2
  BM := 2; BN := 2; BK := 2; GM := 2; SMS := 2

def inputs : RealMath.Inputs where
  a r _ := r + 1
  b k c := k + c + 1

def allocation : FlatAlloc := FlatAlloc.ofList "flat" [("A",24),("B",28),("C",24)]

def initial : BlockState where
  mem r o := if r = "A" then MemCell.real ((o/9 : Nat) + 1)
    else if r = "B" then MemCell.real ((o/11 : Nat) + ((o%11)/2 : Nat) + 1)
    else MemCell.of .int 37
  regs := fun _ _ _ => none
  pids ax := if ax = 0 then 1 else 0
  numPids ax := if ax = 0 then 2 else 1
  undef := fun _ _ => 0

private theorem ba : ("B" : String) ≠ "A" := by decide
private theorem ca : ("C" : String) ≠ "A" := by decide
private theorem cb : ("C" : String) ≠ "B" := by decide

theorem admissible : RealIO.Admissible sample := by
  refine ⟨by decide,by decide,by decide,by decide,by decide,by decide,by decide,by decide,
    by decide,by decide,?_⟩
  rintro ⟨r,c⟩ ⟨r',c'⟩ he
  change 7*r.val+2*c.val = 7*r'.val+2*c'.val at he
  have hr := r.isLt; have hc := c.isLt; have hr' := r'.isLt; have hc' := c'.isLt
  change r.val < 3 at hr; change c.val < 3 at hc
  change r'.val < 3 at hr'; change c'.val < 3 at hc'
  have er : r.val = r'.val := by omega
  have ec : c.val = c'.val := by omega
  exact Prod.ext (Fin.ext er) (Fin.ext ec)

theorem launch : RealIO.Launch sample initial := by
  constructor <;> decide

theorem bounds : RealAccess.Bounds sample allocation.extent := by
  constructor <;> intro r c hr hc
  all_goals norm_num [sample,allocation,FlatAlloc.ofList,FlatAlloc.listExtent,ba,ca,cb] at hr hc ⊢
  all_goals omega

theorem pins : RealInput.Represents sample inputs initial := by
  constructor <;> intro r c hr hc
  all_goals change r < 3 at hr; change c < 3 at hc
  all_goals interval_cases r <;> interval_cases c <;> norm_num [sample,inputs,initial,ba,ca,cb]

theorem disjoint : allocation.Disjoint :=
  FlatAlloc.ofList_disjoint _ _ (by decide)

/-- An explicit nonempty finite-flat execution of both source variants and
both erased storage profiles; no input/output pin or allocation is assumed. -/
theorem executes (storage : TypedSource.Storage) (v : Bool) :
    ∃ t, exec (allocation.flattenKernel (sample.source storage v).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      RealIO.Results sample allocation inputs initial t :=
  RealFull.storage_correct sample storage v admissible allocation initial inputs disjoint rfl
    (FlatAlloc.ofList_closed _ _) rfl launch bounds pins

/-- The flat cells really contain 18/27/36, while the named result retains
input preservation and every raw-cell frame clause. -/
theorem values (storage : TypedSource.Storage) (v : Bool) :
    ∃ t, exec (allocation.flattenKernel (sample.source storage v).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.mem "flat" 66 = MemCell.real 18 ∧
      t.mem "flat" 68 = MemCell.real 27 ∧
      t.mem "flat" 70 = MemCell.real 36 ∧
      RealIO.Results sample allocation inputs initial t := by
  obtain ⟨t,he,hy⟩ := executes storage v
  have h14 := hy.output 0 (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by constructor <;> decide)
  have h16 := hy.output 0 (⟨0,by decide⟩,⟨1,by decide⟩,PUnit.unit) (by constructor <;> decide)
  have h18 := hy.output 1 (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit) (by constructor <;> decide)
  refine ⟨t,he,?_,?_,?_,hy⟩
  all_goals norm_num [initial,RealIO.outputWindow,RealIO.expected,RealStore.offset,
    RealStore.rows,RealStore.columns,RealLoop.tileRow,RealLoop.tileColumn,RealGroup.rows,
    RealGroup.columns,RealMath.blocks,Schedule.row,Schedule.column,Schedule.groupBase,
    Schedule.groupSize,Schedule.tileId,sample,allocation,FlatAlloc.ofList,FlatAlloc.addr,
    FlatAlloc.listBase,RealMath.formula,inputs,Finset.sum_range_succ,ba,ca,cb] at h14 h16 h18
  · exact h14
  · exact h16
  · exact h18

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealFiniteWitness
