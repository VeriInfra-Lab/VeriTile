import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelFP

namespace VeriTile.Bench.Repairs.VarLenCopy.KernelFP
open VeriTile.Triton FP.Structural Source
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Ready (B : Nat) (length oldStart newStart : Int) (s : State α) : Prop where
  lengthReg : s.regs .int [] "length" = some (fun _ => length)
  oldReg : s.regs .int [] "old_start" = some (fun _ => oldStart)
  newReg : s.regs .int [] "new_start" = some (fun _ => newStart)
  oldOffset : s.regs .nat [B] "old_offset" = some (fun i => i.1.val)
  newOffset : s.regs .nat [B] "new_offset" = some (fun i => i.1.val)

def copied (M : Algebra α) {B : Nat} (s : State α) (src : RegionName)
    (length start current : Int) : Values α .real [B] := fun i =>
  M.fp32Load (if current + (i.1.val : Int) < length then
    (s.mem src (start + current + i.1.val).toNat).read .real
    else s.undef .real src (start + current + i.1.val).toNat)

def scatter {B : Nat} (s : State α) (dst : RegionName)
    (length start current : Int) (v : Values α .real [B]) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if current + (i.1.val : Int) < length then
      t.write dst (start + current + i.1.val).toNat (.mk .real (v i)) else t) s

theorem scatter_regs {B : Nat} (s : State α) (dst : RegionName)
    (length start current : Int) (v : Values α .real [B]) :
    (scatter s dst length start current v).regs = s.regs := by
  unfold scatter
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split
    · exact ih _
    · exact ih _

theorem body_run (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) (length oldStart newStart current : Int)
    (h : Ready B length oldStart newStart s)
    (hI : s.regs .int [] "i" = some (fun _ => current)) :
    run M (body Bool.false os ol ns src dst B) s = some
      (scatter (B := B) (s.setReg "v" .real [B] (copied M (B := B) s src length oldStart current))
        dst length newStart current (copied M (B := B) s src length oldStart current)) := by
  simp [body,Source.kernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,h.lengthReg,h.oldReg,h.newReg,h.oldOffset,h.newOffset,hI,
    numeric,numericLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    store,copied,scatter,Region.cast,Option.bind,ComputeDType.eraseDType]
  rfl

theorem body_ready (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) (length oldStart newStart current : Int)
    (h : Ready B length oldStart newStart s) :
    ∃ next, run M (body Bool.false os ol ns src dst B)
      (s.setReg "i" .int [] (fun _ => current)) = some next ∧
      Ready B length oldStart newStart next := by
  have ready : Ready B length oldStart newStart (s.setReg "i" .int [] (fun _ => current)) :=
    ⟨by simpa using h.lengthReg, by simpa using h.oldReg, by simpa using h.newReg,
     by simpa using h.oldOffset, by simpa using h.newOffset⟩
  refine ⟨_,body_run M os ol ns src dst B _ length oldStart newStart current ready (by simp),?_⟩
  constructor <;> rw [scatter_regs]
  · simpa using h.lengthReg
  · simpa using h.oldReg
  · simpa using h.newReg
  · simpa using h.oldOffset
  · simpa using h.newOffset

theorem loop_success (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length oldStart newStart : Int) :
    ∀ cur (s : State α), Ready B length oldStart newStart s →
      ∃ final, rangeInt M "i" cur length B (body Bool.false os ol ns src dst B) s = some final := by
  intro cur s
  induction hh : (length-cur).toNat using Nat.strong_induction_on generalizing cur s with
  | _ remaining ih =>
    intro ready
    by_cases hz : B = 0
    · exact ⟨s,by rw [rangeInt,if_pos hz]⟩
    by_cases active : cur < length
    · obtain ⟨next,he,hn⟩ := body_ready M os ol ns src dst B s length oldStart newStart cur ready
      obtain ⟨final,hf⟩ := ih (length-(cur+B)).toNat (by omega) (cur+B) next rfl hn
      exact ⟨final,by rw [rangeInt,if_neg hz,if_pos active,he]; exact hf⟩
    · exact ⟨s,by rw [rangeInt,if_neg hz,if_neg active]⟩

def prefixState (B : Nat) (os ol ns : Region .int) (s : State α) : State α :=
  let t := s.setReg "a_id" .int [] (fun _ => s.pids 0)
  let t := t.setReg "length" .int [] (fun _ => (s.mem ol.cast (s.pids 0)).read .int)
  let t := t.setReg "old_start" .int [] (fun _ => (s.mem os.cast (s.pids 0)).read .int)
  let t := t.setReg "new_start" .int [] (fun _ => (s.mem ns.cast (s.pids 0)).read .int)
  let t := t.setReg "old_offset" .nat [B] (fun i => i.1.val)
  t.setReg "new_offset" .nat [B] (fun i => i.1.val)

theorem prefix_run (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    run M (prefixBody os ol ns src dst B) s = some (prefixState B os ol ns s) := by
  simp [prefixBody,Source.kernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Region.cast,prefixState]

theorem source_success (M : Algebra α) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (s : State α) :
    ∃ final, exec M (kernel Bool.false os ol ns src dst B) s = some final := by
  let l := (s.mem ol.cast (s.pids 0)).read .int
  let o := (s.mem os.cast (s.pids 0)).read .int
  let n := (s.mem ns.cast (s.pids 0)).read .int
  have ready : Ready B l o n (prefixState B os ol ns s) := by
    constructor <;> simp [prefixState,l,o,n]
  obtain ⟨final,he⟩ := loop_success M os ol ns src dst B l o n 0 (prefixState B os ol ns s) ready
  refine ⟨final,?_⟩
  rw [source_decomposition]
  simp only [FP.Structural.exec,run_append,prefix_run,Option.bind_some,run,step,evalOp_unfold,
    ready.lengthReg,Option.bind_some]
  simpa using he

#axiomsClean source_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.KernelFP
