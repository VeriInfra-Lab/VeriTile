import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Context
import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.FusedRwkv6Kernel_Execution

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
variable {α : Type} [Inhabited α]

/-- Finite stores with a common independent target. This is a memory lemma,
with no numerical hypothesis and no appeal to computed final memory. -/
theorem fold_target {ι β : Type} (lanes : List ι) (active : ι → Bool)
    (name : RegionName) (address : ι → Nat) (value : ι → β) (target : Nat → β)
    (agrees : ∀ i∈lanes,active i=Bool.true → value i=target (address i))
    (m : RegionName → Nat → β) (r : RegionName) (b : Nat) :
    (lanes.foldl (fun current i => if active i then
      fun r b => if r=name ∧ b=address i then value i else current r b
      else current) m) r b =
      if ∃ i∈lanes,active i=Bool.true ∧ r=name ∧ b=address i then target b else m r b := by
  induction lanes generalizing m with
  | nil => simp
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih (fun j hj => agrees j (List.mem_cons_of_mem _ hj))]
    by_cases tail : ∃ j∈rest,active j=Bool.true ∧ r=name ∧ b=address j
    · simp [tail]
    · by_cases ai : active i=Bool.true
      · by_cases hit : r=name ∧ b=address i
        · have hv := agrees i (by simp) ai
          simp [tail,ai,hit,hv,hit.2]
        · simp [tail,ai,hit]
      · simp [tail,ai]

theorem output_address (c : Shape) (s : State α) (n : Nat) (i : TileIndex [c.BV]) :
    outputBase c s i.1+visit c n*c.V=
      partialAddress c (program s).z (program s).ik (visit c n) (valueIndex c (program s) i.1.val) := by
  simp [outputBase,partialAddress,program,valueIndex,Shape.Z,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem region_o (r : FusedRwkv6Kernel.Region) : regionName r="o" ↔ r=.o := by
  cases r <;> simp [regionName]
theorem region_ht (r : FusedRwkv6Kernel.Region) : regionName r="ht" ↔ r=.ht := by
  cases r <;> simp [regionName]

theorem row_hit (c : Shape) (s : State α) (n : Nat) (r : FusedRwkv6Kernel.Region) (b : Nat) :
    (∃ i∈TileShape.allIndices [c.BV],valueMask c s i.1=Bool.true ∧ regionName r="o" ∧
      b=outputBase c s i.1+visit c n*c.V) ↔ PartialWrite c (program s) n r b := by
  constructor
  · rintro ⟨i,_,ha,hr,hb⟩
    refine ⟨(region_o r).mp hr,i.1.val,i.1.isLt,?_,?_⟩
    · simpa [valueMask,valueIndex,program] using ha
    · exact hb.trans (output_address c s n i)
  · rintro ⟨hr,l,hl,ha,hb⟩
    let i : TileIndex [c.BV] := (⟨l,hl⟩,())
    refine ⟨i,TileShape.mem_allIndices _ i,?_,(region_o r).mpr hr,?_⟩
    · simpa [valueMask,valueIndex,program,i] using ha
    · exact hb.trans (output_address c s n i).symm

/-- Readback of the actual DSL row scatter equals the already-proved logical
masked store once the reduction primitive has been instantiated. -/
theorem row_model (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat)
    (plan : Plans c) (m : RegionName → Nat → Cell α)
    (values : ∀ i : TileIndex [c.BV],rowValues M c s scale n i=
      (ops M).storeCast (scalarPartial (ops M) c plan (inputs c (memory s)) scale
        (program s).z (program s).ik n (valueIndex c (program s) i.1.val)))
    (r : FusedRwkv6Kernel.Region) (b : Nat) :
    (rowWrite M c s scale n m (regionName r) b).read .real=
      storePartial (ops M) c (program s) n
        (fun l => scalarPartial (ops M) c plan (inputs c (memory s)) scale
          (program s).z (program s).ik n (valueIndex c (program s) l))
        (fun r b => (m (regionName r) b).read .real) r b := by
  let base:=partialAddress c (program s).z (program s).ik (visit c n) ((program s).iv*c.BV)
  let target : Nat → Cell α := fun b => .mk .real ((ops M).storeCast
    (scalarPartial (ops M) c plan (inputs c (memory s)) scale (program s).z (program s).ik n
      (valueIndex c (program s) (b-base))))
  have hv : ∀ i∈TileShape.allIndices [c.BV],valueMask c s i.1=Bool.true →
      Cell.mk .real (rowValues M c s scale n i)=target (outputBase c s i.1+visit c n*c.V) := by
    intro i _ _
    rw [values i,output_address]
    simp only [target,base,partial_lane]
  unfold rowWrite
  rw [fold_target _ _ "o" _ _ target hv m (regionName r) b]
  simp only [row_hit c s n r b]
  by_cases hw : PartialWrite c (program s) n r b
  · simp only [hw,if_true,Cell.read_same,target,base,storePartial]
    have h := hw
    unfold PartialWrite at h
    rw [if_pos h]
  · simp only [hw,if_false,storePartial]
    have h := hw
    unfold PartialWrite at h
    rw [if_neg h]

end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Source
