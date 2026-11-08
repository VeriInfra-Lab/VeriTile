import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryWhole
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (BackwardConfig DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 700000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- The actual DQ helper keeps all five caller-computed indices intact. -/
structure CallerFrame (s t : State α) : Prop where
  head : t.regs .int [] "i_bh"=s.regs .int [] "i_bh"
  chunk : t.regs .int [] "i_c"=s.regs .int [] "i_c"
  key : t.regs .int [] "i_k"=s.regs .int [] "i_k"
  value : t.regs .int [] "i_v"=s.regs .int [] "i_v"
  decay : t.regs .int [] "i_h"=s.regs .int [] "i_h"

theorem suffix_frame {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s t u : State α) (last : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : Ready M c s t last acc)
    (bounds : HelperIndexBounds c) (ran : run (withDot M D) (suffix c dq) t=some u) :
    CallerFrame s u := by
  simp [suffix,source,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,store,evalOffsets,numeric,bop,withDot,h.queryGrad,
    h.argument0,    h.argument1,    h.argument2,    h.argument3,    h.argument4,    h.argument5,    h.argument6,    h.argument7,    h.argument8,    h.argument9,    h.argument10,    h.argument11,    h.argument12,    h.argument13,
    bounds.query_cast,bounds.key_cast,bounds.value_cast,Index32.natural_product,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat] at ran
  cases ran
  constructor <;> simp [h.outerHead,h.outerChunk,h.outerKey,h.outerValue,h.outerDecay]

def tailCode (c : BackwardConfig) (dq : RegionName) := (source c dq).surfaceBody.drop 28

theorem tail_split (c : BackwardConfig) (dq : RegionName) :
    tailCode c dq=numericalPrefix c dq++[firstRange c]++middle c dq++[secondRange c]++suffix c dq:=rfl

theorem tail_run {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s captured : State α)
    (capturedArgs : ArgumentReady c s captured) (bounds : HelperIndexBounds c) :
    ∃ final,run (withDot M D) (tailCode c dq) captured=some final ∧
      final.mem=(outputMemory M c dq s (values M D c s)).mem ∧
      final.undef=s.undef ∧ CallerFrame s final := by
  obtain ⟨initial,hp,hi⟩:=numerical_run M D c dq s captured capturedArgs bounds
  obtain ⟨first,hr,last,hf⟩:=first_range_run M D c s initial 0 (c.i_c*c.BTL) _ hi
  have hrZero : rangeInt (withDot M D) "_" 0 ((c.i_c*c.BTL : Nat) : Int) c.BTS (firstBody c) initial=some first := by
    simpa only [Int.natCast_zero] using hr
  obtain ⟨diagonal,hm,hd⟩:=middle_run M D c dq s first last _ hf bounds
  obtain ⟨second,hs,finish,hz⟩:=second_range_run M D c s diagonal (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd bounds.lanes
  obtain ⟨final,he,hout,hu⟩:=suffix_run M D c dq s second finish _ hz bounds
  refine ⟨final,?_,?_,hu,suffix_frame M D c dq s second final finish _ hz bounds he⟩
  · rw [tail_split]
    simp only [run_append,hp,bind,pure,Option.bind,run,
      first_range_statement M D c initial hi.argument1,hrZero,hm,
      second_range_statement M D c diagonal hd.argument1,hs,he]
  · simpa only [values] using hout

#axiomsClean suffix_frame
#axiomsClean tail_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPQuery
