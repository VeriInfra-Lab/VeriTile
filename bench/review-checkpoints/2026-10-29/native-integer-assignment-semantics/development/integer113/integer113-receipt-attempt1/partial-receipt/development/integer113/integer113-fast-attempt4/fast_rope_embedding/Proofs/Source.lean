import bench.repairs.tritonbench_g.fast_rope_embedding.Kernels
import bench.repairs.tritonbench_g.rope_embedding.Proofs.Contract

/-! Fast RoPE really assigns ROPE_GROUP_SIZE in its body. The other RoPE
accepts that value as a constexpr parameter. Prove this source's complete
prelude and signed loop directly; the independent mathematical helpers are shared.
The two whole kernels are not claimed to be syntactically equal. -/
namespace VeriTile.Bench.Repairs.FastRopeEmbedding.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

export VeriTile.Bench.Repairs.RopeEmbedding.Source
  (Parameters Lane active offset coefficient Context value written heads
    heads_mem_congr heads_readback heads_frame groupResult written_mem_congr)

/-- The unchanged four-head body, including both loaded halves, both stores,
the backwards sign, load-to-sine casts and FP32 output conversion. -/
def source (p : Parameters) (backward : Bool) : ComputeKernel :=
  Kernels.rope p.input p.cosine p.sine
    p.rowStride p.cosineStride p.sineStride p.length p.dimension p.heads p.width backward
def prelude (p : Parameters) (backward : Bool) := (source p backward).surfaceBody.take 11
def body (p : Parameters) (backward : Bool) : List ComputeStmt :=
  match (source p backward).surfaceBody.drop 11 with
  | [.forRangeInt _ _ _ _ stmts] => stmts
  | _ => []

theorem decomposition (p : Parameters) (backward : Bool) :
    (source p backward).surfaceBody = prelude p backward ++
      [.forRangeInt "k" (.ref .int [] "head_start") (.ref .int [] "head_end") (.constNat 1)
        (body p backward)] := by
  cases backward <;> rfl

def prepared {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool) (s : State α) :
    {v : State α // run M (prelude p backward) s=some v} := by
  cases backward <;>
    (refine ⟨?_,?_⟩
     swap
     simp [prelude,source,Kernels.rope,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
       Region.cast,Option.bind,integralFloorDiv,integralMod]
     try rfl)

theorem prepared_facts {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) :
    Context p (s.pids 0) (coefficient M p s p.cosine p.cosineStride)
      (fun i => if backward then M.unary (some .fp32) (.neg .real) (coefficient M p s p.sine p.sineStride i)
        else coefficient M p s p.sine p.sineStride i) (prepared M p backward s).val ∧
    (prepared M p backward s).val.mem=s.mem ∧
    (prepared M p backward s).val.regs .int [] "head_start"=some (fun _ => ((s.pids 1*4 : Nat) : Int)) ∧
    (prepared M p backward s).val.regs .int [] "head_end"=some (fun _ => ((min (s.pids 1*4+4) p.heads : Nat) : Int)) := by
  cases backward <;> refine ⟨?_,rfl,?_,?_⟩
  all_goals try constructor
  all_goals simp [prepared,coefficient,active,State.setReg,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,integralMod,min_def_lt,-Nat.cast_mul,-Nat.cast_add,←Int.natCast_mul,←Int.natCast_add,←Int.natCast_ediv]
  all_goals first | rfl | (funext i; split_ifs <;> omega)

def oneHead {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool)
    (row head : Nat) (cs ss : Lane p → α) (s : State α) (h : Context p row cs ss s) :
    {v : State α // run M (body p backward) (s.setReg "k" .int [] (fun _ => (head : Int)))=some v} := by
  cases backward <;>
    (refine ⟨?_,?_⟩
     swap
     simp [body,source,Kernels.rope,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
       Region.cast,Option.bind,State.setReg,h.row,h.columns,h.half,h.mask,h.cosine,h.sine,
       -Nat.cast_mul,-Nat.cast_add,←Int.natCast_mul,←Int.natCast_add,←Int.natCast_ediv]
     try rfl)

theorem oneHead_facts {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool)
    (row head : Nat) (cs ss : Lane p → α) (s : State α) (h : Context p row cs ss s) :
    Context p row cs ss (oneHead M p backward row head cs ss s h).val ∧
    (oneHead M p backward row head cs ss s h).val.mem=(written M p row head cs ss s).mem := by
  constructor
  · cases backward <;> constructor <;> simp [oneHead,State.setReg,h.row,h.columns,h.half,h.mask,h.cosine,h.sine,
       -Nat.cast_mul,-Nat.cast_add,←Int.natCast_mul,←Int.natCast_add,←Int.natCast_ediv]
  · cases backward <;> simp +contextual [oneHead,written,RotaryEmbNopad.Pair.written,
      RotaryEmbNopad.Pair.halfScatter,masked_scatter_memory,value,offset,active,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,State.setReg,Region.cast,
      h.row,h.columns,h.half,h.mask,h.cosine,h.sine,
       -Nat.cast_mul,-Nat.cast_add,←Int.natCast_mul,←Int.natCast_add,←Int.natCast_ediv]

theorem loop_executes {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (backward : Bool)
    (row : Nat) (cs ss : Lane p → α) (start count : Nat) (s : State α) (h : Context p row cs ss s) :
    ∃ final, rangeInt M "k" (start : Int) ((start+count : Nat) : Int) 1 (body p backward) s=some final ∧
      Context p row cs ss final ∧ final.mem=(heads M p row cs ss start count s).mem := by
  induction count generalizing start s with
  | zero =>
    refine ⟨s,?_,h,rfl⟩
    rw [rangeInt]
    simp
  | succ count ih =>
    let v := (oneHead M p backward row start cs ss s h).val
    have hv := (oneHead M p backward row start cs ss s h).property
    have hf := oneHead_facts M p backward row start cs ss s h
    obtain ⟨final,he,hc,hm⟩ := ih (start+1) v hf.1
    refine ⟨final,?_,hc,?_⟩
    · rw [rangeInt]
      simp only [show (1 : Nat) ≠ 0 by decide,↓reduceIte,
        show (start : Int) < ((start+(count+1) : Nat) : Int) by omega,hv,Option.bind_some]
      simpa only [Int.natCast_add, Int.natCast_one, Int.add_assoc, Int.add_left_comm, Int.add_comm] using he
    · exact hm.trans (heads_mem_congr M p row cs ss (start+1) count v
        (written M p row start cs ss s) hf.2)

theorem complete_execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) (group : s.pids 1*4≤p.heads) :
    ∃ final, FP.Structural.exec M (source p backward) s=some final ∧
      final.mem=(groupResult M p backward s).mem := by
  obtain ⟨hc,hm,hs,he⟩ := prepared_facts M p backward s
  have first : s.pids 1*4≤min (s.pids 1*4+4) p.heads := le_min (by omega) group
  have stop : s.pids 1*4+(min (s.pids 1*4+4) p.heads-s.pids 1*4)=min (s.pids 1*4+4) p.heads :=
    Nat.add_sub_of_le first
  obtain ⟨final,hloop,_,hmem⟩ := loop_executes M p backward (s.pids 0) _ _
    (s.pids 1*4) (min (s.pids 1*4+4) p.heads-s.pids 1*4) (prepared M p backward s).val hc
  refine ⟨final,?_,?_⟩
  · change run M (source p backward).surfaceBody s=some final
    rw [decomposition,run_append,(prepared M p backward s).property]
    simp only [Option.bind_some,run,step,evalOp_unfold,hs,he,Option.bind]
    simp only [bind,Option.bind]
    rw [← stop,hloop]
  · exact hmem.trans (heads_mem_congr M p (s.pids 0) _ _ _ _ _ s hm)


theorem complete_correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (backward : Bool) (s : State α) (group : s.pids 1*4≤p.heads) :
    ∃ final,FP.Structural.exec M (source p backward) s=some final ∧
      (∀ h,s.pids 1*4≤h → h<min (s.pids 1*4+4) p.heads → ∀ side i,active p i →
        final.mem p.input (offset p (s.pids 0) h side i)=.mk .real
          (value M p (s.pids 0) h (coefficient M p s p.cosine p.cosineStride)
            (fun j => if backward then M.unary (some .fp32) (.neg .real) (coefficient M p s p.sine p.sineStride j)
              else coefficient M p s p.sine p.sineStride j) s side i)) ∧
      (∀ region o,(region≠p.input ∨ ∀ h,s.pids 1*4≤h → h<min (s.pids 1*4+4) p.heads →
        ∀ side i,active p i → o≠offset p (s.pids 0) h side i) → final.mem region o=s.mem region o) := by
  obtain ⟨final,he,hm⟩ := complete_execution M p backward s group
  have hs : s.pids 1*4≤min (s.pids 1*4+4) p.heads := le_min (by omega) group
  refine ⟨final,he,?_,?_⟩
  · intro h hlo hhi side i hi
    rw [hm]
    exact heads_readback M p (s.pids 0) _ _ _ _ s h hlo (by simpa [Nat.add_sub_of_le hs] using hhi) side i hi
  · intro region o outside
    rw [hm]
    exact heads_frame M p (s.pids 0) _ _ _ _ s region o
      (outside.imp id (fun h k hlo hhi => h k hlo (by simpa [Nat.add_sub_of_le hs] using hhi)))


#axiomsClean oneHead_facts
#axiomsClean loop_executes
#axiomsClean complete_execution
#axiomsClean complete_correctness
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.Source
