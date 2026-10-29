import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardHValues

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

abbrev Accum (c : Shape) (α : Type) := (Index c → α) × State α
def advance (M : Algebra α) (D : DotPrimitive α) (c : Shape) (initial : State α)
    (n : Nat) (a : Accum c α) : Accum c α :=
  ⟨update M D c initial n a.1,stored M c initial a.2 n a.1⟩
def accumulated (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed : Bool)
    (initial : State α) : Accum c α :=
  foldRange (advance M D c initial) 0 c.chunks 1 ⟨initValue M c seed initial,initial⟩

theorem body_parts (c : Shape) (seed final : Bool) :
    (parts c seed final).body=prelude c seed final++suffix c seed final :=
  (List.take_append_drop 4 (parts c seed final).body).symm

theorem loopExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (ctx : Context M c initial current 0)
    (ha : HasCarry c (initValue M c seed initial) current) (hm : current.mem=initial.mem) :
    ∃ t, range (engine (withDot M D)) "i_t" 0 c.chunks 1 (parts c seed final).body current=some t ∧
      (∃ n, Context M c initial t n) ∧ HasCarry c (accumulated M D c seed initial).1 t ∧
      t.mem=(accumulated M D c seed initial).2.mem := by
  let I := fun (n : Nat) (a : Accum c α) (s : State α) =>
    Context M c initial s n ∧ HasCarry c a.1 s ∧ s.mem=a.2.mem
  let Done := fun (a : Accum c α) (s : State α) =>
    (∃ n, Context M c initial s n) ∧ HasCarry c a.1 s ∧ s.mem=a.2.mem
  apply range_values (engine (withDot M D)) "i_t" (parts c seed final).body
    (advance M D c initial) I Done 0 c.chunks 1 _ current
  · intro n a s h hn
    let prep := preludeExecution M D c seed final initial s n a.1 h.1 h.2.1
    let hp := preludePrepared M D c seed final initial s n a.1 h.1 h.2.1
    let finish := suffixExecution M D c seed final initial prep.val n a.1 hn hp
    refine ⟨finish.val,?_,suffixContext M D c seed final initial prep.val n a.1 hn hp,
      suffixCarry M D c seed final initial prep.val n a.1 hn hp,?_⟩
    · simp only [body_parts,run_append,prep.property,Option.bind_some]
      exact finish.property
    · rw [suffixMemory,preludeMemory]
      simp only [advance,stored,masked_scatter_memory,h.2.2]
  · intro n a s h
    exact ⟨⟨n,h.1⟩,h.2⟩
  · exact ⟨ctx,ha,hm⟩

def finalStored (M : Algebra α) (c : Shape) (initial current : State α) (carry : Index c → α) : State α :=
  (TileShape.allIndices [c.BK,c.BV]).foldl (fun t i =>
    if active c initial i then t.write "ht" (stateAddress c initial i)
      (.mk .real (M.cast (some .fp32) .real .real (carry i))) else t) current
def written (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool) (initial : State α) : State α :=
  let a := accumulated M D c seed initial
  if final then finalStored M c initial a.2 a.1 else a.2

def finishExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat) (carry : Index c → α)
    (ctx : Context M c initial current n) (ha : HasCarry c carry current) :
    {t : State α // run (engine (withDot M D)) (parts c seed final).after current=some t} := by
  simp only [HasCarry] at ha
  cases final
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Kernels.forwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,
      Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
      resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,numericEq,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      ctx.ik,ctx.iv,ctx.bh,ha,FP.Structural.store]
    rfl

theorem finishMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool)
    (initial current : State α) (n : Nat)
    (ctx : Context M c initial current n)
    (ha : HasCarry c (accumulated M D c seed initial).1 current)
    (hm : current.mem=(accumulated M D c seed initial).2.mem) :
    (finishExecution M D c seed final initial current n (accumulated M D c seed initial).1 ctx ha).val.mem=
      (written M D c seed final initial).mem := by
  cases final <;>
    simp [finishExecution,written,finalStored,masked_scatter_memory,active,stateAddress,feature,column,
      bop,BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,hm]

theorem source_parts (c : Shape) (seed final : Bool) : (kernel c seed final).surfaceBody =
    (parts c seed final).before ++
    [.forRangeDyn "i_t" (parts c seed final).start (parts c seed final).stop
      (parts c seed final).stride (parts c seed final).body] ++ (parts c seed final).after := rfl

theorem source_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (seed final : Bool) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel c seed final) s=some t ∧
      t.mem=(written M D c seed final s).mem := by
  let start := entryExecution M D c seed final s
  have hc := entryContext M D c seed final s
  have hmem : start.val.mem=s.mem := by cases seed <;> simp [start,entryExecution]
  obtain ⟨mid,hm,⟨n,ctx⟩,ha,hmm⟩ := loopExecution M D c seed final s start.val hc.1 hc.2 hmem
  let finish := finishExecution M D c seed final s mid n (accumulated M D c seed s).1 ctx ha
  refine ⟨finish.val,?_,finishMemory M D c seed final s mid n ctx ha hmm⟩
  have hl : step (engine (withDot M D)) (.forRangeDyn "i_t" (parts c seed final).start (parts c seed final).stop
      (parts c seed final).stride (parts c seed final).body) start.val=some mid := by
    have h0 : (parts c seed final).start = .constNat 0 := rfl
    have hn : (parts c seed final).stop = .constNat c.chunks := rfl
    have h1 : (parts c seed final).stride = .constNat 1 := rfl
    simp only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop]
    exact hm
  change run (engine (withDot M D)) (kernel c seed final).surfaceBody s=some finish.val
  rw [source_parts,List.append_assoc,run_append,start.property]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]

#specNonCircular written avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardHSource
