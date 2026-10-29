import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Source
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Lowering

namespace VeriTile.Bench.Repairs.IsfiniteKernel.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 24000

/-- Unwritten output bytes retain their complete byte value and Nat tag;
only actual Bool stores encode a canonical zero or one. -/
def cellByte {A : Type} : Cell A → BitVec 8
  | .mk .bool b => Memory.boolByte b
  | .mk .nat n => BitVec.ofNat 8 n
  | _ => 0

variable {M A : Type} [Inhabited A] {width : Nat} (ops : Memory.Primitives M A width)

def backed (before : M) (pid : Nat) : State A where
  mem r q := if r=("x" : RegionName) then .mk .real (ops.read before q) else .mk .nat (ops.byte before q).toNat
  regs _ _ _ := none
  pids _ := pid
  numPids _ := 1
  undef d _ _ := defaultValue d

@[simp] theorem backed_read (before : M) (pid q : Nat) :
    ((backed ops before pid).mem "x" q).read .real=ops.read before q := by simp [backed,Cell.read]
@[simp] theorem backed_pid (before : M) (pid : Nat) : (backed ops before pid).pids 0=pid := rfl
@[simp] theorem backed_byte (before : M) (pid q : Nat) :
    cellByte ((backed ops before pid).mem "y" q)=ops.byte before q := by simp [backed,cellByte]

theorem fold_projection {ι : Type} (xs : List ι) (active : ι → Bool)
    (index output : Nat → Nat) (lane : ι → Nat) (values : Nat → Bool)
    (s : State A) (before : M)
    (relation : ∀ q,cellByte (s.mem "y" q)=ops.byte before q) :
    ∀ q,cellByte ((xs.foldl (fun t i => if active i then
      t.write "y" (output (index (lane i))) (.mk .bool (values (index (lane i)))) else t) s).mem "y" q)=
      ops.byte (Memory.stores ops output values ((xs.filter active).map (fun i => index (lane i))) before) q := by
  induction xs generalizing s before with
  | nil => exact relation
  | cons i rest ih =>
    cases h : active i
    · simpa [h] using ih s before relation
    · simp only [List.foldl_cons,h,↓reduceIte,List.filter_cons_of_pos h,List.map_cons,Memory.stores]
      apply ih
      intro q
      rw [ops.write_byte]
      by_cases equal : q=output (index (lane i))
      · simp [State.write,equal,cellByte]
      · simp [State.write,equal,relation q]

theorem index_values (B : Nat) :
    (TileShape.allIndices [B]).map (fun i => i.1.val)=List.range B := by
  simp only [TileShape.allIndices,List.map_flatMap,List.map_cons,List.map_nil,List.map_map]
  change (List.finRange B).flatMap (fun i => [i.val])=List.range B
  rw [←List.map_eq_flatMap,List.map_coe_finRange_eq_range]

private theorem filtered_map {ι β : Type} (xs : List ι) (f : ι → β) (p : β → Bool) :
    (xs.filter (fun i => p (f i))).map f=(xs.map f).filter p := by
  induction xs with
  | nil => rfl
  | cons i rest ih => cases h : p (f i) <;> simp [h,ih]

theorem source_jobs (start count pid : Nat) (small : count<2^31) (launched : pid<Finite.grid count) :
    ((TileShape.allIndices [1024]).filter (fun i => decide (pid*1024+i.1.val<count))).map
      (fun i => start+(pid*1024+i.1.val))=Lowering.jobs start count pid := by
  have masks : (List.range 1024).filter (fun lane => decide (pid*1024+lane<count))=
      (List.range 1024).filter (Finite.mask count pid) := by
    apply List.filter_congr
    intro lane member
    exact (VeriTile.Bench.TritonBenchG.VectorAddition.Native.bounded_arithmetic count pid lane
      small launched (List.mem_range.mp member)).2.symm
  have mapped := filtered_map (TileShape.allIndices [1024]) (fun i => i.1.val)
    (fun lane => decide (pid*1024+lane<count))
  rw [index_values,masks] at mapped
  simpa only [List.map_map,Function.comp_def,Lowering.jobs,F8ConversionUtils.Lowering.jobs,
    F8ConversionUtils.Lowering.lanes] using congrArg (List.map (fun lane => start+(pid*1024+lane))) mapped

theorem stores_projection (classify : A → Bool) (input output : View)
    (start count B pid : Nat) (before : M) :
    ∀ q,cellByte ((stores classify (input.shift start) (output.shift start) count B pid
      (backed ops before pid) (backed ops before pid)).mem "y" q)=
      ops.byte (Memory.stores ops output.address (fun i => classify (ops.read before (input.address i)))
        (((TileShape.allIndices [B]).filter (fun i => decide (pid*B+i.1.val<count))).map
          (fun i => start+(pid*B+i.1.val))) before) q := by
  have projection := fold_projection ops (TileShape.allIndices [B])
    (fun i => decide (pid*B+i.1.val<count)) (fun lane => start+(pid*B+lane))
    output.address (fun i => i.1.val) (fun i => classify (ops.read before (input.address i)))
    (backed ops before pid) before (backed_byte ops before pid)
  simpa only [stores,backed_read,Finite.shifted_address,decide_eq_true_eq] using projection

/-- Every byte of the source output region equals the literal native program
on the same pre-program heap. The input region is reconstructed from that
heap for every call, including same-allocation disjoint strided views. -/
theorem program_refines (classify : A → Bool) (x y : ViewMetadata) (h : InputContract x y)
    (start count pid : Nat) (small : count<2^31) (launched : pid<Finite.grid count)
    (globalActive : ∀ i,i<count → start+i<x.n) (before : M) :
    ∃ final,FP.Structural.exec (algebra x.precision classify)
        (source x.precision (x.view.shift start) (y.outputView.shift start) count 1024)
        (backed ops before pid)=some final ∧
      (∀ q,cellByte (final.mem "y" q)=ops.byte
        (Lowering.program ops classify (x.view.shift start) (y.outputView.shift start) count pid before) q) ∧
      (∀ r q,r≠("y" : RegionName) → final.mem r q=(backed ops before pid).mem r q) := by
  obtain ⟨final,executed,result⟩ := executes x.precision classify (x.view.shift start)
    (y.outputView.shift start) count 1024 (backed ops before pid)
  refine ⟨final,executed,?_,?_⟩
  · intro q
    rw [result,Lowering.program_eq ops classify x y h start count pid small launched globalActive]
    change cellByte ((stores classify (x.view.shift start) (y.outputView.shift start) count 1024 pid
      (backed ops before pid) (backed ops before pid)).mem "y" q)=
      ops.byte (Memory.stores ops y.outputView.address (fun i => classify (ops.read before (x.view.address i)))
        (Lowering.jobs start count pid) before) q
    have projection := stores_projection ops classify x.view y.outputView start count 1024 pid before q
    rw [source_jobs start count pid small launched] at projection
    exact projection
  · intro r q different
    rw [result]
    unfold stores
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (Or.inl different) _

#axiomsClean program_refines
end
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.IsfiniteKernel.Source
