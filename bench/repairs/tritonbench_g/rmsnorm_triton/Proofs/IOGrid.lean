import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.IORow

namespace VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton
open Whole KernelGrid
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def rowAddress (g : Geometry) (b m k : Nat) : Nat :=
  b*g.output.stride 0+m*g.output.stride 1+k*g.output.stride 2

 theorem input_frame (g : Geometry) (s t : BlockState) (vals : (io g).Ctx)
    (inputs : (io g).RegionalInputs s vals)
    (frame : ∀ (r : RegionName) address, r≠"out" → t.mem r address=s.mem r address) :
    (io g).RegionalInputs t vals := by
  intro i j active
  have v := inputs i j active
  change t.readMem _ _=vals i j
  change s.readMem _ _=vals i j at v
  rw [BlockState.readMem,frame _ _ (by
    dsimp only [io]
    split <;> decide)]
  exact v

 theorem grid_run (g : Geometry) (unique : OutputUnique g.parameters)
    (jobs : List (Nat×Nat)) (valid : ∀ bm ∈ jobs, bm.1<g.batches ∧ bm.2<g.rows)
    (s : BlockState) (vals : (io g).Ctx) (inputs : (io g).RegionalInputs s vals) :
    ∃ final,
      runAlgorithmSchedule (io g).kernel.toAlgKernel
        (jobs.map (fun bm => launch bm.1 bm.2)) s=some final ∧
      (∀ q : Fin (rowCount g), (batch g q.val,row g q.val)∈jobs →
        ∀ k : Fin g.width, final.readMem "out" (outputAddress g q.val k.val)=target g vals q k) ∧
      (∀ (r : RegionName) address,
        (∀ bm ∈ jobs, r≠"out" ∨ ∀ k : Fin g.width,
          address≠rowAddress g bm.1 bm.2 k.val) → final.mem r address=s.mem r address) := by
  induction jobs generalizing s with
  | nil => exact ⟨s,rfl,by simp,fun _ _ _ => rfl⟩
  | cons bm rest ih =>
    have hbm := valid bm (by simp)
    let q := index g bm.1 bm.2 hbm.1 hbm.2
    obtain ⟨t,ran,values,frame⟩ := row_run g unique s vals inputs bm.1 bm.2 hbm.1 hbm.2
    have ti : (io g).RegionalInputs t vals :=
      input_frame g s t vals inputs (fun r address different => frame r address (Or.inl different))
    obtain ⟨final,ranRest,restValues,restFrame⟩ :=
      ih (fun other member => valid other (by simp [member])) t ti
    refine ⟨final,?_,?_,?_⟩
    · simp only [List.map_cons,runAlgorithmSchedule,ran,Option.bind_some]
      exact ranRest
    · intro out member k
      by_cases done : (batch g out.val,row g out.val)∈rest
      · exact restValues out done k
      have atHead : (batch g out.val,row g out.val)=bm := by
        simpa only [List.mem_cons,done,or_false] using member
      have sameIndex : out=q := by
        apply Fin.ext
        have h := congrArg Prod.fst atHead
        have h' := congrArg Prod.snd atHead
        change out.val=bm.1*g.rows+bm.2
        rw [←h,←h']
        exact (Nat.div_add_mod' out.val g.rows).symm
      have untouched : final.mem "out" (outputAddress g out.val k.val)=
          t.mem "out" (outputAddress g out.val k.val) := by
        apply restFrame
        intro other otherMem
        right
        intro j equal
        have ho := valid other (by simp [otherMem])
        let r := index g other.1 other.2 ho.1 ho.2
        have eq' : outputAddress g out.val k.val=outputAddress g r.val j.val := by
          simpa [r,outputAddress,rowAddress] using equal
        have qr := (unique_coordinates g unique out r k j eq').1
        apply done
        simpa [qr,r] using otherMem
      rw [BlockState.readMem,untouched]
      change t.readMem "out" (outputAddress g out.val k.val)=target g vals out k
      exact sameIndex.symm ▸ values k
    · intro r address outside
      rw [restFrame r address (fun other member => outside other (by simp [member]))]
      apply frame
      simpa [outputAddress,rowAddress] using outside bm (by simp)

 theorem complete_run (g : Geometry) (unique : OutputUnique g.parameters)
    (work : Schedule g.parameters) (s : BlockState) (vals : (io g).Ctx)
    (inputs : (io g).RegionalInputs s vals) :
    ∃ final, runAlgorithmSchedule (io g).kernel.toAlgKernel (schedule g work vals) s=some final ∧
      (∀ (out : Fin (io g).nOut) (k : Fin ((io g).oarity out)),
        (io g).omask out vals (s.pids 0) (s.pids 1) (s.pids 2) k →
        ((io g).oty out).read final ((io g).obuf out)
          ((io g).owin out vals (s.pids 0) (s.pids 1) (s.pids 2) k)=target g vals out k) ∧
      (io g).RegionalFrame s final vals := by
  obtain ⟨final,ran,values,frame⟩ := grid_run g unique work.order
    (fun bm member => member_bounds g work bm.1 bm.2 member) s vals inputs
  refine ⟨final,ran,?_,?_⟩
  · intro out k _
    apply values out
    exact work.complete.mem_iff.mpr ((programGrid_mem g.parameters _ _).mpr
      ⟨batch_lt g out,row_lt g out⟩)
  · intro r address outside _
    apply frame r address
    intro bm member
    have hb := member_bounds g work bm.1 bm.2 member
    let q := index g bm.1 bm.2 hb.1 hb.2
    by_cases hr : r="out"
    · right
      intro k
      have h := outside q k trivial
      simpa [io,q,outputAddress,rowAddress,hr] using h
    · exact Or.inl hr

#axiomsClean complete_run
end
end VeriTile.Bench.Repairs.RmsnormTriton.RmsIO
