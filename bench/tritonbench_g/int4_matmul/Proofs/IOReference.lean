import bench.tritonbench_g.int4_matmul.Proofs.IOOperands
import bench.tritonbench_g.int4_matmul.Proofs.IOSchedule

namespace VeriTile.Bench.Repairs.Int4Matmul.IOProof
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

/-- Filtering complete source lanes per address preserves the old exact native
recurrence, including arbitrary atomic split order and inactive launch lanes. -/
theorem reference_order (s : Shape) (t : Tile) (o : Ops α)
    (v : ChannelValues α (IO.inputs s)) (lanes : List Lane)
    (valid : ∀ l∈lanes,l.Valid s t) (r c : Nat) (hr : r<s.m) (hc : c<s.n) (seed : α) :
    referenceOutput s t o (operands s o v) r c (laneSchedule s t lanes) seed=
      (IO.order s t lanes r c).foldl (fun acc p => if t.splits=1 then
        Formula.partialSum s t o v r c p.2 else
        o.atomicAdd acc (Formula.partialSum s t o v r c p.2)) seed := by
  induction lanes generalizing seed with
  | nil => rfl
  | cons l lanes ih =>
    have tail : ∀ k∈lanes,k.Valid s t := fun k hk => valid k (by simp [hk])
    by_cases matchRC : (laneWork s t l).row=r ∧ (laneWork s t l).col=c
    · have active : s.Active t (laneWork s t l) :=
        ⟨matchRC.1 ▸ hr,matchRC.2 ▸ hc,(valid l (by simp)).2.1⟩
      have partialEq : referencePartial s t o (operands s o v) (laneWork s t l)=
          Formula.partialSum s t o v r c l.split := by
        rw [formula_partial,matchRC.1,matchRC.2]
        rfl
      simpa only [laneSchedule,List.filter_cons,decide_eq_true active,ite_true,List.map_cons,
        referenceOutput,matchRC,ite_true,IO.order,List.filter_cons,decide_eq_true matchRC,
        List.map_cons,List.foldl_cons,partialEq] using ih tail _
    · by_cases active : s.Active t (laneWork s t l)
      · simpa only [laneSchedule,List.filter_cons,decide_eq_true active,ite_true,List.map_cons,
          referenceOutput,matchRC,ite_false,IO.order,List.filter_cons,decide_eq_false matchRC,
          Bool.false_eq_true,ite_false] using ih tail seed
      · simpa only [laneSchedule,List.filter_cons,decide_eq_false active,Bool.false_eq_true,ite_false,
          IO.order,List.filter_cons,decide_eq_false matchRC] using ih tail seed

theorem real_formula (x : Raw.Arguments) (geometry : Raw.Geometry x) (t : Tile)
    (member : t∈Host.sourceTiles) (lanes : List Lane) (complete : lanes.Perm (rawLanes x.shape t))
    (v : ChannelValues ℝ (IO.inputs x.shape)) (r c : Nat) (hr : r<x.shape.m) (hc : c<x.shape.n) :
    Formula.observed x.shape t realOps v r c (IO.order x.shape t lanes r c)=Formula.mathematical x.shape v r c := by
  have ref := reference_order x.shape t realOps v lanes
    (fun l hl=>(raw_mem x.shape t l).mp (complete.mem_iff.mp hl)) r c hr hc realOps.zero
  change _=Formula.observed x.shape t realOps v r c (IO.order x.shape t lanes r c) at ref
  have complete' := KernelExecution.schedule_complete x.shape t lanes complete
  have value := whole_real x.shape t geometry.shape (Host.all_tiles_valid t member)
    (operands x.shape realOps v) (laneSchedule x.shape t lanes) complete' r c hr hc
  rw [call_result x.shape t geometry.shape (Host.all_tiles_valid t member) realOps
    (laneSchedule x.shape t lanes) complete' (operands x.shape realOps v) r c hr hc] at value
  rw [←ref]
  exact value

#axiomsClean reference_order
#axiomsClean real_formula
end
end VeriTile.Bench.Repairs.Int4Matmul.IOProof
