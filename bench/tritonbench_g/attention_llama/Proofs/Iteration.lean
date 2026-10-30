import bench.tritonbench_g.attention_llama.Proofs.ReachedBytes

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Ordinary signed-byte storage observation. Values outside described views
may be filled with zero; neither byte finiteness nor a numerical result is used. -/
def ByteRange (s : State (Term α)) : Prop :=
  ∀ r o, Float8E5Bits.isInt8 ((s.mem r o).read .int)=Bool.true

 theorem masked_bytes {shape : TileShape} (s : State (Term α)) (h : ByteRange s)
    (p : TileIndex shape → RegionName × Nat) (mask : TileIndex shape → Bool) :
    (TileShape.allIndices shape).all (fun i=>Float8E5Bits.isInt8
      (if mask i then (s.mem (p i).1 (p i).2).read .int else 0))=Bool.true := by
  simp only [List.all_eq_true]
  intro i _
  split
  · exact h _ _
  · decide

theorem masked_cell (s : State (Term α)) (h : ByteRange s)
    (r : RegionName) (o : Nat) (p : Prop) [Decidable p] :
    Float8E5Bits.isInt8 (if p then (s.mem r o).read .int else 0)=Bool.true := by
  split
  · exact h r o
  · decide

theorem ByteRange.reached (c : Config) (s : State (Term α)) (h : ByteRange s) :
    ReachedBytes c s := ⟨fun _ _ _ _=>h _ _,fun _ _ _ _=>h _ _⟩

def iteration_of_reached (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ReachedBytes c s) :
    {u : State (Term α) // run algebra (parts c).body
      (t.setReg "start_n" .nat [] (fun _=>n*64))=some u} := by
  cases hf : c.fp8
  ·
    cases hc : c.causal
    all_goals
      refine ⟨?_,?_⟩
      swap
      simp [parts,kernel,hf,hc,TypedSource.half,TypedSource.bytes,
        ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
        ComputeDType.eraseDType,
        numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
        TileShape.axisDim,Region.cast,State.setReg,
        h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
        h.batch,h.head,h.key,h.value,h.memory,h.pids]
      rfl
  · have hk : ∀ (a : Fin 64) (b : Fin c.D), Float8E5Bits.isInt8
        (if n*64+a.val<c.N then
          (s.mem c.K (pointers c s c.K c.skz c.skh c.skn c.skk n (a,b,())).2).read .int else 0)=Bool.true :=
      fun a b=>masked_key_cell c s bytes n (a,b,()) hf
    have hv : ∀ (a : Fin 64) (b : Fin c.D), Float8E5Bits.isInt8
        (if n*64+a.val<c.N then
          (s.mem c.V (pointers c s c.V c.svz c.svh c.svk c.svn n (a,b,())).2).read .int else 0)=Bool.true :=
      fun a b=>masked_value_cell c s bytes n (a,b,()) hf
    cases hc : c.causal
    all_goals
      refine ⟨?_,?_⟩
      swap
      simp [parts,kernel,hf,hc,TypedSource.half,TypedSource.bytes,
        ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
        run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,
        ComputeDType.eraseDType,
        numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
        TileShape.axisDim,Region.cast,State.setReg,
        h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
        h.batch,h.head,h.key,h.value,h.memory,h.pids,hk,hv]
      rfl

def iteration (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) (bytes : ByteRange s) :
    {u : State (Term α) // run algebra (parts c).body
      (t.setReg "start_n" .nat [] (fun _=>n*64))=some u} :=
  iteration_of_reached c s t scale n a h (bytes.reached c s)

#axiomsClean iteration_of_reached
#axiomsClean iteration
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
