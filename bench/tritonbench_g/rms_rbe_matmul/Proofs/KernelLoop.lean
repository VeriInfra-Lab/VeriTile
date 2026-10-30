import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelPrefix

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem source_zero : (0.0 : ℝ)=0 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num

/-- Signed-byte validity at exactly the active reads of one source K block.
The FP16 branch has no signed-byte premise. -/
def BlockBytes (fp8 : Bool) (p : Parameters) (s : State α) (cur : Nat) : Prop :=
  fp8=Bool.true →
    (∀ i : TileIndex [1,64], i.2.1.val<p.K-cur*64 →
      Float8E5Bits.isInt8 ((s.mem "RMS" (rOffset cur i)).read .int)=Bool.true) ∧
    (∀ i : TileIndex [64,64], i.1.val<p.K-cur*64 →
      Float8E5Bits.isInt8 ((s.mem "W" (wOffset p s cur i)).read .int)=Bool.true)

theorem blockBytes_of_byteMemory (fp8 : Bool) (p : Parameters) (s : State α)
    (cur : Nat) (bytes : ByteMemory s) : BlockBytes fp8 p s cur := by
  intro _
  exact ⟨fun _ _ => bytes "RMS" (.inr rfl) _,fun _ _ => bytes "W" (.inl rfl) _⟩

theorem body_run_reads (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (cur : Nat) (acc : Accumulator α)
    (bytes : BlockBytes fp8 p s cur) (h : Ready p s t cur acc) :
    ∃ u, run (engine a) (loop fp8 p)
      (t.setReg "step" .nat [] (fun _ => cur))=some u ∧
      Ready p s u (cur+1) (update a fp8 p s cur acc) := by
  have hr : fp8=Bool.false ∨ (TileShape.allIndices [1,64]).all (fun i => Float8E5Bits.isInt8
      (if i.2.1.val<p.K-cur*64 then (s.mem "RMS" (rOffset cur i)).read .int else 0))=Bool.true := by
    cases hf : fp8
    · exact Or.inl rfl
    · right
      apply List.all_eq_true.mpr
      intro i _
      split
      · exact (bytes hf).1 i ‹_›
      · decide
  have hw : fp8=Bool.false ∨ (TileShape.allIndices [64,64]).all (fun i => Float8E5Bits.isInt8
      (if i.1.val<p.K-cur*64 then (s.mem "W" (wOffset p s cur i)).read .int else 0))=Bool.true := by
    cases hf : fp8
    · exact Or.inl rfl
    · right
      apply List.all_eq_true.mpr
      intro i _
      split
      · exact (bytes hf).2 i ‹_›
      · decide
  cases fp8
  all_goals simp only [Bool.false_eq_true, Bool.true_eq_false, false_or, true_or] at hr hw
  all_goals
    simp [-List.all_cons,-List.all_nil,-List.all_map,-List.all_flatMap,-List.all_eq_true,
      loop,kernel,TypedSource.kernel,RmsMatmulRbe.TypedSource.kernel,arguments,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      engine,Algebra.withDefaultPrecision,resolvePrecision,
      h.mem,h.pids,h.undef,h.batch,h.tileM,h.tileN,h.offk,h.xptr,h.wptr,h.rptr,h.product,h.squares,
      xPointer,wPointer,rPointer,hr,hw]
  all_goals constructor
  all_goals try simp [-List.all_cons,-List.all_nil,-List.all_map,-List.all_flatMap,-List.all_eq_true,
    State.setReg,h.mem,h.pids,h.undef,h.batch,h.tileM,h.tileN,h.offk,
    h.xptr,h.wptr,h.rptr,h.product,h.squares,update,xValue,rmsHalf,wHalf,scaledX,weight,
    zero,halfZero,xPointer,wPointer,rPointer,xOffset,wOffset,rOffset,bop,ofFloat,toFloat]
  all_goals first | rfl | (funext i; simp [bop,Nat.add_mul,Nat.add_assoc,
    xPointer,wPointer,rPointer,xOffset,wOffset,rOffset])
  all_goals rcases i with ⟨i,j,⟨⟩⟩
  all_goals simp [scaledX,weight,xValue,rmsHalf,wHalf,zero,halfZero,xOffset,wOffset,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,source_zero,Nat.add_assoc]
  all_goals congr 2
  all_goals funext q
  all_goals rcases q with ⟨q,k,⟨⟩⟩
  all_goals simp [scaledX,weight,xValue,rmsHalf,wHalf,zero,halfZero,xOffset,wOffset,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,source_zero,Nat.add_assoc]

/-- Original all-offset endpoint retained as a specialization. -/
theorem body_run (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (cur : Nat) (acc : Accumulator α)
    (bytes : ByteMemory s) (h : Ready p s t cur acc) :
    ∃ u, run (engine a) (loop fp8 p)
      (t.setReg "step" .nat [] (fun _ => cur))=some u ∧
      Ready p s u (cur+1) (update a fp8 p s cur acc) :=
  body_run_reads a fp8 p s t cur acc (blockBytes_of_byteMemory fp8 p s cur bytes) h

theorem loop_run_reads (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (start stop : Nat) (acc : Accumulator α)
    (bytes : ∀ cur, start≤cur → cur<stop → BlockBytes fp8 p s cur)
    (h : Ready p s t start acc) :
    ∃ u, range (engine a) "step" start stop 1 (loop fp8 p) t=some u ∧
      Ready p s u (max start stop)
        (foldRange (update a fp8 p s) start stop 1 acc) := by
  by_cases active : start<stop
  · obtain ⟨v,ran,ready⟩ := body_run_reads a fp8 p s t start acc (bytes start (by omega) active) h
    obtain ⟨u,rest,final⟩ := loop_run_reads a fp8 p s v (start+1) stop
      (update a fp8 p s start acc) (fun cur lower upper => bytes cur (by omega) upper) ready
    refine ⟨u,?_,?_⟩
    · rw [range]
      simp [active,ran,rest]
    · rw [foldRange]
      simpa only [show (1:Nat)≠0 by decide,active,if_false,if_true,
        Nat.max_eq_right (by omega : start≤stop),
        Nat.max_eq_right (by omega : start+1≤stop)] using final
  · refine ⟨t,?_,?_⟩
    · rw [range]
      simp [active]
    · rw [foldRange]
      simpa only [show (1:Nat)≠0 by decide,active,if_false,
        Nat.max_eq_left (by omega : stop≤start)] using h
termination_by stop-start
decreasing_by omega

/-- Original full-memory premise and complete loop conclusion are preserved. -/
theorem loop_run (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (start stop : Nat) (acc : Accumulator α)
    (bytes : ByteMemory s) (h : Ready p s t start acc) :
    ∃ u, range (engine a) "step" start stop 1 (loop fp8 p) t=some u ∧
      Ready p s u (max start stop)
        (foldRange (update a fp8 p s) start stop 1 acc) :=
  loop_run_reads a fp8 p s t start stop acc
    (fun cur _ _ => blockBytes_of_byteMemory fp8 p s cur bytes) h

#axiomsClean body_run_reads
#axiomsClean loop_run_reads

#axiomsClean body_run
#axiomsClean loop_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
