import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPReference

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Replica32
open VeriTile.Triton FP.Structural
open Native
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000

/-- Saved SM80 B32/C32 inner-backward butterfly, with dynamic N. The address
within its logical row selects the physical lane that consumes the reduction. -/
def lane (N : Nat) (consumer : Consumer) : Fin 32 := ⟨(consumer.2%N)%32, Nat.mod_lt _ (by decide)⟩
def partner (i : Fin 32) (distance : Nat) : Fin 32 :=
  ⟨(Nat.xor i.val distance)%32,Nat.mod_lt _ (by decide)⟩

def tree : Nat → Fin 32 → Scalar.Tree 32
  | 0,i => .lane i
  | k+1,i => .join (tree k i) (tree k (partner i (2^(4-k))))

theorem all_leaves (i : Fin 32) : (tree 5 i).leaves.Perm ((List.finRange 32).map some) := by
  fin_cases i <;> decide

def plan (i : Fin 32) : Scalar.Plan 32 := ⟨tree 5 i,0,by simpa using all_leaves i⟩

/-- Only the reached 32-lane axis is this saved compilation. The fallback
exists to make the algebra total and is not asserted to be another compilation.
All other public profiles use the same consumer-indexed interface. -/
def reductions (N : Nat) : Reductions := fun consumer _ _ s ax _ _ _ =>
  if h : TileShape.axisDim s ax=32 then h.symm ▸ plan (lane N consumer)
  else SoftmaxFlaggemsFiniteSeed.Native.Plan.seeded (TileShape.axisDim s ax)

def scalarSchedule : Scalar.Schedule where
  fusion := fun sh => match sh with
    | .binary (some .fp32) .real .add (.binary (some .fp32) .real .mul _ _) _ => .left
    | _ => .split
  swapMultiply := fun sh => match sh with
    | .binary (some .fp32) .real .mul _ (.binary (some .fp32) .real .sub _ _) => Bool.true
    | _ => Bool.false

def schedule : Schedule := fun _ => scalarSchedule

/-- Structural binding of the actual source scale to the selected consumer
replica. Source masks and FP32 load barriers remain in the leaf expressions. -/
theorem scale_binding (N row rows : Nat) (P G : RegionName) (base : State Term)
    (r : Fin 32) (consumer : Consumer) :
    SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerScale (algebra (reductions N)) base P G row rows N 32 32 (r,PUnit.unit) consumer=
      (tree 5 (lane N consumer)).expand (some .fp32) .add
        (fun n => .binary (some .fp32) .real .mul
          (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerLoad (algebra (reductions N)) base P row rows N 32 32 (r,n,PUnit.unit) consumer)
          (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerLoad (algebra (reductions N)) base G row rows N 32 32 (r,n,PUnit.unit) consumer))
        (.literal (some .fp32) .real 0) := by
  rfl

/-- The first stage is FMA(local product, rounded partner product), in this
ordered operand form. Opposite consumers need not have the same result. -/
theorem first_stage (scalar : ℝ → Word)
    (p g : Fin 32 → Scalar.Term) (i : Fin 32) :
    Scalar.lower scalar scalarSchedule
      ((tree 1 i).expand (some .fp32) .add
        (fun j => .binary (some .fp32) .real .mul (p j) (g j)) (.literal (some .fp32) .real 0))=
      Ksoftmax.Graph.Expr.fma (Scalar.lower scalar scalarSchedule (p i)) (Scalar.lower scalar scalarSchedule (g i))
        (Scalar.lower scalar scalarSchedule (.binary (some .fp32) .real .mul (p (partner i 16)) (g (partner i 16)))) := by
  rfl

/-- These are different ordered replicas even though they cover the same row. -/
theorem opposite_first_pairs : partner (0 : Fin 32) 16=(16 : Fin 32) ∧ partner (16 : Fin 32) 16=(0 : Fin 32) := by decide


/-- The physical lane selected by a valid source store is its n coordinate. -/
theorem store_lane (N row r n : Nat) (Y : RegionName) (hn : n<N) (hB : n<32) :
    lane N (Y,(row*32+r)*N+n)=⟨n,hB⟩ := by
  apply Fin.ext
  simp only [lane,Nat.mul_add_mod_of_lt hn,Nat.mod_eq_of_lt hB]

/-- Every remaining PTX shuffle/add stage retains the corresponding ordered
replicas. The recurrence starts with distance16 and finishes with 8,4,2,1. -/
theorem remaining_stage (scalar : ℝ → Word) (p g : Fin 32 → Scalar.Term)
    (i : Fin 32) (k : Nat) :
    Scalar.lower scalar scalarSchedule
      ((tree (k+2) i).expand (some .fp32) .add
        (fun j => .binary (some .fp32) .real .mul (p j) (g j)) (.literal (some .fp32) .real 0))=
      Ksoftmax.Graph.Expr.add
        (Scalar.lower scalar scalarSchedule ((tree (k+1) i).expand (some .fp32) .add
          (fun j => .binary (some .fp32) .real .mul (p j) (g j)) (.literal (some .fp32) .real 0)))
        (Scalar.lower scalar scalarSchedule ((tree (k+1) (partner i (2^(4-(k+1))))).expand (some .fp32) .add
          (fun j => .binary (some .fp32) .real .mul (p j) (g j)) (.literal (some .fp32) .real 0))) := by
  rfl

/-- The source store consumes this lane's scale, then emits sub followed by
multiply in the saved PTX operand order. There is no common-row result node. -/
theorem stored_binding (scalar : ℝ → Word) (N row rows : Nat) (P G : RegionName)
    (base : State Term) (i : TileIndex [32,32]) (consumer : Consumer) :
    NativeProfile.lower scalar schedule consumer
      (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerValue (algebra (reductions N)) base P G row rows N 32 32 i)=
      Ksoftmax.Graph.Expr.mul
        (.sub
          (NativeProfile.lower scalar schedule consumer
            (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerLoad (algebra (reductions N)) base G row rows N 32 32 i))
          (Scalar.lower scalar scalarSchedule
            ((tree 5 (lane N consumer)).expand (some .fp32) .add
              (fun n => .binary (some .fp32) .real .mul
                (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerLoad (algebra (reductions N)) base P row rows N 32 32 (i.1,n,PUnit.unit) consumer)
                (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerLoad (algebra (reductions N)) base G row rows N 32 32 (i.1,n,PUnit.unit) consumer))
              (.literal (some .fp32) .real 0))))
        (NativeProfile.lower scalar schedule consumer
          (SoftmaxFlaggemsFiniteSeed.FPReference.backwardInnerLoad (algebra (reductions N)) base P row rows N 32 32 i)) := by
  rfl

#axiomsClean store_lane
#axiomsClean remaining_stage
#axiomsClean stored_binding

#axiomsClean all_leaves
#axiomsClean scale_binding
#axiomsClean first_stage
#axiomsClean opposite_first_pairs
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Replica32
