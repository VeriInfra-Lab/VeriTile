import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelFPSuccess

namespace VeriTile.Bench.Repairs.VarLenCopy.KernelFP
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
variable {α : Type} [Inhabited α]

/-- The frame uses only the independent requested destination interval. -/
def Footprint (dst : RegionName) (length newStart : Int) (r : RegionName) (a : Nat) : Prop :=
  r = dst ∧ ∃ i : Int, 0 ≤ i ∧ i < length ∧ a = (newStart+i).toNat

theorem scatter_frame {B : Nat} (s : State α) (dst : RegionName)
    (length newStart current : Int) (v : Values α .real [B]) (hc : 0 ≤ current)
    (r : RegionName) (a : Nat) (outside : ¬Footprint dst length newStart r a) :
    (scatter s dst length newStart current v).mem r a = s.mem r a := by
  unfold scatter
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split
    next active =>
      rw [ih]
      simp only [State.write]
      split
      next eq =>
        rcases eq with ⟨hr,ha⟩
        exact False.elim (outside ⟨hr, current + i.1.val, by omega, active, by simpa [Int.add_assoc] using ha⟩)
      next => rfl
    next => exact ih _

theorem loop_frame (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length oldStart newStart : Int) :
    ∀ cur (s final : State α), 0 ≤ cur → Ready B length oldStart newStart s →
      rangeInt M "i" cur length B (body Bool.false os ol ns src dst B) s = some final →
      ∀ r a, ¬Footprint dst length newStart r a → final.mem r a = s.mem r a := by
  intro cur s final
  induction hh : (length-cur).toNat using Nat.strong_induction_on generalizing cur s final with
  | _ remaining ih =>
    intro hc ready he r a outside
    by_cases hz : B = 0
    · rw [rangeInt,if_pos hz] at he
      cases he
      rfl
    by_cases active : cur < length
    · obtain ⟨next,hbody,hn⟩ := body_ready M os ol ns src dst B s length oldStart newStart cur ready
      rw [rangeInt,if_neg hz,if_pos active,hbody] at he
      have htail := ih (length-(cur+B)).toNat (by omega) (cur+B) next final rfl (by omega) hn he r a outside
      rw [htail]
      have hi : Ready B length oldStart newStart (s.setReg "i" .int [] (fun _ => cur)) :=
        ⟨by simpa using ready.lengthReg,by simpa using ready.oldReg,by simpa using ready.newReg,
         by simpa using ready.oldOffset,by simpa using ready.newOffset⟩
      rw [body_run M os ol ns src dst B _ length oldStart newStart cur hi (by simp)] at hbody
      cases hbody
      exact scatter_frame _ dst length newStart cur _ hc r a outside
    · rw [rangeInt,if_neg hz,if_neg active] at he
      cases he
      rfl

theorem source_frame (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s final : State α)
    (he : exec M (kernel Bool.false os ol ns src dst B) s = some final) :
    ∀ r a, ¬Footprint dst ((s.mem ol.cast (s.pids 0)).read .int)
      ((s.mem ns.cast (s.pids 0)).read .int) r a → final.mem r a = s.mem r a := by
  let l := (s.mem ol.cast (s.pids 0)).read .int
  let o := (s.mem os.cast (s.pids 0)).read .int
  let n := (s.mem ns.cast (s.pids 0)).read .int
  have ready : Ready B l o n (prefixState B os ol ns s) := by
    constructor <;> simp [prefixState,l,o,n]
  rw [source_decomposition] at he
  simp only [FP.Structural.exec,run_append,prefix_run,Option.bind_some,run,step,evalOp_unfold,
    ready.lengthReg,Option.bind_eq_bind,Option.bind_some] at he
  exact loop_frame M os ol ns src dst B l o n 0 (prefixState B os ol ns s) final (by omega) ready (by simpa using he)

#axiomsClean source_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.KernelFP
