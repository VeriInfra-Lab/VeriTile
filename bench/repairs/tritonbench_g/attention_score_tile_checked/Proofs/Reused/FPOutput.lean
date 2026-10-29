import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPLoop
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPEngine FPContext
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def active (B : Nat) (p : RealSource.Parameters) (s : State α) (i : TileIndex [B]) : Prop :=
  s.pids 0*B+i.1.val<p.keyContext
instance (B : Nat) (p : RealSource.Parameters) (s : State α) : DecidablePred (active B p s) :=
  fun _ => inferInstanceAs (Decidable (_<_))
def address (B : Nat) (p : RealSource.Parameters) (s : State α) (i : TileIndex [B]) : Nat :=
  s.pids 1/p.heads*p.outputBatch+s.pids 1%p.heads*p.outputHead+s.pids 0*B+i.1.val

def value (M : Algebra α) (acc : Fin B → α) (i : TileIndex [B]) : α :=
  M.cast (some .fp32) .real .real (acc i.1)

def executes (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) :
    {t : State α // run (engine (withDot M dot)) (parts B D p).after current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,FPEngine.kernel,RealSource.source,TypedSource.original,Whole002.TypedSource.widened,
    ComputeKernel.surfaceBody,Source.intRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.startN,ctx.batch,ctx.head,ctx.output,FP.Structural.store]
  rfl

def written (M : Algebra α) (B : Nat) (p : RealSource.Parameters) (s : State α) (acc : Fin B → α) : State α :=
  (TileShape.allIndices [B]).foldl (fun t i =>
    if active B p s i then t.write p.Out (address B p s i) (.mk .real (value M acc i)) else t) s

theorem memory (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) :
    (executes M dot B D p initial current row acc ctx).val.mem=(written M B p initial acc).mem := by
  simp [executes,written,masked_scatter_memory,ctx.memory,active,address,value,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  simp only [← Int.natCast_add, ← Int.natCast_mul, ← Int.natCast_ediv,
    ← Int.natCast_emod, Int.toNat_natCast, Int.ofNat_lt]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem source_parts (B D : Nat) (p : RealSource.Parameters) :
    (FPEngine.kernel B D p).surfaceBody=(parts B D p).before ++
      [.forRangeInt "start_m" (parts B D p).start (parts B D p).stop (parts B D p).stride (parts B D p).body] ++
      (parts B D p).after := rfl

/-- The entire actual typed source runs and returns precisely the finite masked
stores of the independently defined ordered primitive recurrence. -/
theorem source_memory (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α)
    (keyOffsetBound : s.pids 0*B < 2147483648) :
    ∃ t, FP.Structural.exec (engine (withDot M dot)) (FPEngine.kernel B D p) s=some t ∧
      t.mem=(written M B p s (FPLoop.accumulated M dot B D p s)).mem := by
  let start := FPEntry.executes M dot B D p s
  obtain ⟨mid,hm,row,hc⟩ := FPLoop.executes M dot B D p s start.val (entry_context M dot B D p s keyOffsetBound)
  let finish := executes M dot B D p s mid row (FPLoop.accumulated M dot B D p s) hc
  refine ⟨finish.val,?_,memory M dot B D p s mid row _ hc⟩
  have hl : step (engine (withDot M dot))
      (.forRangeInt "start_m" (parts B D p).start (parts B D p).stop (parts B D p).stride (parts B D p).body)
      start.val=some mid := by
    have h0 : (parts B D p).start=.ref .int [] "lo" := rfl
    have hn : (parts B D p).stop=.ref .int [] "hi" := rfl
    have hb : (parts B D p).stride=.constNat B := rfl
    have hlo : start.val.regs .int [] "lo"=some (fun _ => 0) := (FPLoop.entry_limits M dot B D p s).1
    have hhi : start.val.regs .int [] "hi"=some (fun _ => (p.queryContext : Int)) := (FPLoop.entry_limits M dot B D p s).2
    simpa only [step,h0,hn,hb,evalOp_unfold,hlo,hhi,Option.bind,Option.bind_some] using hm
  change run (engine (withDot M dot)) (FPEngine.kernel B D p).surfaceBody s=some finish.val
  rw [source_parts,List.append_assoc,run_append,start.property]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]

theorem readback (M : Algebra α) (B : Nat) (p : RealSource.Parameters) (s : State α) (acc : Fin B → α)
    (i : TileIndex [B]) (hi : active B p s i) :
    (written M B p s acc).mem p.Out (address B p s i)=.mk .real (value M acc i) := by
  apply masked_readback_active _ _ _ _ _ i hi
  intro j k _ _ he
  have h : j.1.val=k.1.val := by simpa [address] using he
  rcases j with ⟨j,⟨⟩⟩
  rcases k with ⟨k,⟨⟩⟩
  exact congrArg (fun i : Fin B => (i,PUnit.unit)) (Fin.ext h)

theorem frame (M : Algebra α) (B : Nat) (p : RealSource.Parameters) (s : State α) (acc : Fin B → α)
    (r : RegionName) (off : Nat) (h : r≠p.Out ∨ ∀ i, active B p s i → off≠address B p s i) :
    (written M B p s acc).mem r off=s.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

#specNonCircular value avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPOutput
