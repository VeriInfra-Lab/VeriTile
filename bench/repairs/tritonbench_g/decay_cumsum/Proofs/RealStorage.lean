import bench.repairs.tritonbench_g.decay_cumsum.Proofs.RealIO
import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Run

namespace VeriTile.Bench.Repairs.DecayCumsum.RealStorage
open VeriTile.Triton Geometry Grid PackedStorage
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

abbrev OutputIndex (h : Call) (kind : Kind) (scale : ℝ) (w : Work h) :=
  Fin (RealIO.io kind h w.1 scale).nOut × Fin (RealIO.io kind h w.1 scale).T × Fin (RealIO.io kind h w.1 scale).B

def inputs (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (w : Work h) (heap : Heap ℝ) :=
  fun (i : Fin (RealIO.io kind h w.1 scale).nIn) (t : Fin (RealIO.io kind h w.1 scale).T)
    (j : Fin (RealIO.io kind h w.1 scale).B) =>
    lift storage w.1 heap ((RealIO.io kind h w.1 scale).inp i) (relative h w.1 w.2 t.val j.val)

def execute (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (w : Work h) (heap : Heap ℝ) : Option (Heap ℝ) :=
  (VeriTile.Triton.exec ((RealAlloc.allocation h kind storage w.1).flattenKernel
    ((Kernels.source kind h w.1 scale).eraseDType.toAlgKernel))
      ((RealAlloc.allocation h kind storage w.1).flattenState (RealAlloc.entry h kind storage w heap))).map
    (fun v => commit storage w.1 heap (RealAlloc.pull h kind storage w.1 v))

def outputAddress (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (w : Work h)
    (i : OutputIndex h kind scale w) : Nat :=
  address storage ((RealIO.io kind h w.1 scale).out i.1) (Geometry.offset h w.1 w.2 i.2.1.val i.2.2.val)

def expected (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (heap : Heap ℝ)
    (w : Work h) (i : OutputIndex h kind scale w) : ℝ :=
  RealIO.values kind h w.1 scale (inputs h kind storage scale w heap) i.1 i.2.1 i.2.2

theorem time_bound (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (t : Fin (RealIO.io kind h p scale).T) :
    t.val<p.span := by simpa only [RealIO.timeSize] using t.isLt

theorem lane_bound (kind : Kind) (h : Call) (p : Phase) (scale : ℝ) (j : Fin (RealIO.io kind h p scale).B) :
    j.val<h.BK := by simpa only [RealIO.laneSize] using j.isLt

theorem inside (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (w : Work h) (i : OutputIndex h kind scale w) :
    footprint h kind storage w (outputAddress h kind storage scale w i) :=
  ⟨_,Roles.output_member kind _ (RealIO.output_member kind h w.1 scale i.1),
    ⟨i.2.1.val,time_bound kind h w.1 scale i.2.1⟩,⟨i.2.2.val,lane_bound kind h w.1 scale i.2.2⟩,rfl⟩

theorem complete (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (w : Work h) (member : w∈schedule h) (heap : Heap ℝ) :
    ∃ final, execute h kind storage scale w heap=some final ∧
      (∀ i, final (outputAddress h kind storage scale w i)=expected h kind storage scale heap w i) ∧
      (∀ addr, ¬footprint h kind storage w addr → final addr=heap addr) := by
  let A := RealAlloc.allocation h kind storage w.1
  let u := RealAlloc.entry h kind storage w heap
  let io := RealIO.io kind h w.1 scale
  have bound (r : RegionName) (hr : r∈Roles.buffers kind) (t : Fin io.T) (j : Fin io.B) :
      relative h w.1 w.2 t.val j.val<A.extent r :=
    RealAlloc.extent_bound h kind storage w member r hr
      ⟨t.val,time_bound kind h w.1 scale t⟩ ⟨j.val,lane_bound kind h w.1 scale j⟩
  have run := RealIO.implements kind h w.1 (schedule_valid h w member) scale (RealAlloc.feature_positive h w.1 w.2)
  obtain ⟨v,he,output,frame⟩ := run A (RealAlloc.disjoint h kind storage w.1)
    (RealIO.buffers kind h w.1 scale).symm (by intro r hr; exact if_neg hr)
    w.2.feature w.2.chunk w.2.row (inputs h kind storage scale w heap) u
    (by simp [u,RealAlloc.entry]) (by simp [u,RealAlloc.entry]) (by simp [u,RealAlloc.entry]) rfl
    (by intro i t j _; rw [RealIO.read_offset]; exact bound _ (RealIO.input_member kind h w.1 scale i) t j)
    (by intro i t j _; rw [RealIO.write_offset]; exact bound _ (Roles.output_member kind _ (RealIO.output_member kind h w.1 scale i)) t j)
    (by intro i t j _; rw [RealIO.read_offset]; rfl)
  refine ⟨commit storage w.1 heap (RealAlloc.pull h kind storage w.1 v),?_,?_,?_⟩
  · unfold execute
    rw [←RealIO.source kind h w.1 scale]
    change (VeriTile.Triton.exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState u)).map _=_
    rw [he]
    rfl
  · intro i
    have offsetBound := offset_inside h w.1 (schedule_valid h w member) w.2
      ⟨i.2.1.val,time_bound kind h w.1 scale i.2.1⟩ ⟨i.2.2.val,lane_bound kind h w.1 scale i.2.2⟩
    unfold outputAddress
    rw [←relocated,commit_output h kind storage w.1 heap _ _ (RealIO.output_member kind h w.1 scale i.1) _ (by rwa [relocated])]
    have out := output i.1 i.2.1 i.2.2 (RealIO.write_active kind h w.1 scale w.2 i.1 i.2.1 i.2.2)
    rw [RealIO.write_offset] at out
    exact out
  · intro addr outside
    unfold commit
    cases hd : decode storage w.1 addr with
    | none => rfl
    | some pair =>
      obtain ⟨memberR,inBounds,addrEq⟩ := decode_some h kind storage w.1 addr pair hd
      have hr := Roles.output_member kind pair.1 memberR
      have localBound : pair.2<A.extent pair.1 := by
        change pair.2<(if pair.1∈Roles.buffers kind then h.cells-shift h w.1 else 0)
        rw [if_pos hr]
        omega
      have unchanged := frame A.flat (A.addr pair.1 pair.2) (Or.inr (by
        intro o t j _ eq
        rw [RealIO.write_offset] at eq
        have decoded := congrArg A.decode eq
        rw [A.decode_addr (RealAlloc.disjoint h kind storage w.1) hr localBound,
          A.decode_addr (RealAlloc.disjoint h kind storage w.1)
            (Roles.output_member kind _ (RealIO.output_member kind h w.1 scale o))
            (bound _ (Roles.output_member kind _ (RealIO.output_member kind h w.1 scale o)) t j)] at decoded
        have pairs := Option.some.inj decoded
        have er := congrArg Prod.fst pairs
        have eo := congrArg Prod.snd pairs
        apply outside
        refine ⟨pair.1,hr,⟨t.val,time_bound kind h w.1 scale t⟩,⟨j.val,lane_bound kind h w.1 scale j⟩,?_⟩
        rw [addrEq,eo,relocated]))
      change RealAlloc.pull h kind storage w.1 v pair.1 pair.2=heap addr
      have readEq : RealAlloc.pull h kind storage w.1 v pair.1 pair.2=
          RealAlloc.pull h kind storage w.1 (A.flattenState u) pair.1 pair.2 := by
        unfold RealAlloc.pull BlockState.readMem
        rw [unchanged]
      rw [readEq,RealAlloc.pull_entry h kind storage w heap pair.1 hr pair.2 inBounds]
      exact congrArg heap addrEq.symm

theorem expected_stable (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ)
    (w : Work h) (_member : w∈schedule h) (a b : Heap ℝ)
    (stable : ∀ addr, footprint h kind storage w addr → a addr=b addr) (i : OutputIndex h kind scale w) :
    expected h kind storage scale a w i=expected h kind storage scale b w i := by
  have equal : inputs h kind storage scale w a=inputs h kind storage scale w b := by
    funext r t j
    apply stable
    refine ⟨_,RealIO.input_member kind h w.1 scale r,⟨t.val,time_bound kind h w.1 scale t⟩,
      ⟨j.val,lane_bound kind h w.1 scale j⟩,?_⟩
    congr 1
    exact relocated h w.1 w.2 t.val j.val
  unfold expected
  rw [equal]

theorem grid_complete (h : Call) (kind : Kind) (storage : Placement h kind) (scale : ℝ) (heap : Heap ℝ) :
    ∃ final, Run.run (execute h kind storage scale) (schedule h) heap=some final ∧
      (∀ w∈schedule h, ∀ i, final (outputAddress h kind storage scale w i)=expected h kind storage scale heap w i) ∧
      (∀ addr, (∀ w∈schedule h, ¬footprint h kind storage w addr) → final addr=heap addr) := by
  exact Run.complete (execute h kind storage scale) (OutputIndex h kind scale) (outputAddress h kind storage scale) id
    (expected h kind storage scale) (footprint h kind storage) (inside h kind storage scale) (schedule h)
    (schedule_nodup h) (fun a ha b hb => disjoint h kind storage a b ha hb) (complete h kind storage scale)
    (expected_stable h kind storage scale) heap

#axiomsClean grid_complete
end
end VeriTile.Bench.Repairs.DecayCumsum.RealStorage
