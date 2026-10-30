import bench.tritonbench_g.chunk_linear_attn.Frame
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe foldRange_last scatter_readback)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem history_write_read (M : Algebra α) (p : Parameters) (pid : PID) (rev : Bool)
    (n : Nat) (h : HValues p α) (m : Memory α) (i : TileIndex [p.BK,p.BV])
    (hn : n<p.NT) (hi : hActive p pid i) :
    writeHistory M p pid rev n h m (historyName rev) (hOffset p pid (chunk p rev n) i)=.mk .real (cast M (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (Geometry.history_collision p pid pid _ _ a b ha hb (Geometry.chunk_lt p rev n hn)
      (Geometry.chunk_lt p rev n hn) he).2.2)

theorem final_write_read (M : Algebra α) (p : Parameters) (pid : PID) (rev : Bool)
    (h : HValues p α) (m : Memory α) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    writeStateFinal M p pid rev Bool.true h m (finalName rev) (fOffset p pid i)=.mk .real (cast M (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he => (Geometry.final_collision p pid pid a b ha hb he).2)

theorem state_read_history (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (rev seed final : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV])
    (hn : n<p.NT) (hi : hActive p pid i) :
    stateMemory M D p pid input m scale rev seed final (historyName rev) (hOffset p pid (chunk p rev n) i)=
      .mk .real (cast M (stateValues M D p pid input scale rev seed n i)) := by
  have hf : ∀ mem, writeStateFinal M p pid rev final (stateValues M D p pid input scale rev seed p.NT)
      mem (historyName rev) (hOffset p pid (chunk p rev n) i)=mem (historyName rev) (hOffset p pid (chunk p rev n) i) := by
    intro mem
    cases final
    · rfl
    · apply scatter_frame
      left
      cases rev <;> decide
  unfold stateMemory
  rw [hf]
  apply foldRange_last _ (fun mem : Memory α => mem (historyName rev) (hOffset p pid (chunk p rev n) i)) p.NT n m _ hn
  · intro mem
    exact history_write_read M p pid rev n _ mem i hn hi
  · intro k hnk hk mem
    apply scatter_frame
    right
    intro j _ hj he
    have hc := (Geometry.history_collision p pid pid _ _ i j hi hj
      (Geometry.chunk_lt p rev n hn) (Geometry.chunk_lt p rev k hk) he).2.1
    have hnkeq := Geometry.chunk_injective p rev n k hn hk hc
    omega

theorem state_read_final (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (rev seed : Bool) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    stateMemory M D p pid input m scale rev seed Bool.true (finalName rev) (fOffset p pid i)=
      .mk .real (cast M (stateValues M D p pid input scale rev seed p.NT i)) :=
  final_write_read M p pid rev _ _ i hi

theorem output_read (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (i : TileIndex [p.BT,p.BV]) (hi : outputActive p pid i) :
    outputMemory M D p pid input m scale "output" (outputOffset p pid i)=
      .mk .real (cast M (outputResult M D p pid input scale (outputStates M D p pid input (keyBlocks p)) i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he => (Geometry.output_collision p pid pid a b ha hb he).2)

theorem gradKey_write_read (M : Algebra α) (p : Parameters) (pid : PID) (r : RegionName)
    (a : Matrix α p.BT p.BK) (m : Memory α) (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradKeyWrite M p pid r a m r (gradKeyOffset p pid i)=.mk .real (cast M (a i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he => (Geometry.gradient_collision p pid pid a b ha hb he).2)

theorem gradient_read_query (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradientMemory M D p pid (p.B*p.H) input m scale "queryGradient" (gradKeyOffset p pid i)=
      .mk .real (cast M (gradQueryResult M D p pid input scale (gradStates M D p pid input scale (valueBlocks p)) i)) := by
  unfold gradientMemory
  rw [show ∀ a mem, gradKeyWrite M p pid "keyGradient" a mem "queryGradient" (gradKeyOffset p pid i)=
      mem "queryGradient" (gradKeyOffset p pid i) from fun a mem => scatter_frame _ _ _ _ _ _ _ _ (Or.inl (by decide))]
  exact gradKey_write_read M p pid "queryGradient" _ _ i hi

theorem gradient_read_key (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (i : TileIndex [p.BT,p.BK]) (hi : gradKeyActive p pid i) :
    gradientMemory M D p pid (p.B*p.H) input m scale "keyGradient" (gradKeyOffset p pid i)=
      .mk .real (cast M (gradKeyResult M D p pid input scale (gradStates M D p pid input scale (valueBlocks p)) i)) :=
  gradKey_write_read M p pid "keyGradient" _ _ i hi

theorem gradient_read_value (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (vb : Nat) (i : TileIndex [p.BT,p.BV])
    (hh : pid.z<p.B*p.H) (hv : vb<valueBlocks p) (hi : gradValueActive p pid vb i) :
    gradientMemory M D p pid (p.B*p.H) input m scale "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i)=
      .mk .real (cast M (gradValue M D p pid input scale vb i)) := by
  have hk : ∀ (dest : RegionName), dest≠"valuePartials" → ∀ a mem,
      gradKeyWrite M p pid dest a mem "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i)=
      mem "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i) := by
    intro dest hd a mem
    exact scatter_frame _ _ _ _ _ _ _ _ (Or.inl (Ne.symm hd))
  unfold gradientMemory
  rw [hk "keyGradient" (by decide),hk "queryGradient" (by decide)]
  apply foldRange_last _ (fun mem : Memory α => mem "valuePartials" (gradValueOffset p pid (p.B*p.H) vb i))
    (valueBlocks p) vb m _ hv
  · intro mem
    exact scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
      (Geometry.partial_collision p (p.B*p.H) pid pid vb vb a b hh hh ha hb he).2.2)
  · intro other hvother hother mem
    apply scatter_frame
    right
    intro j _ hj he
    have hb := (Geometry.partial_collision p (p.B*p.H) pid pid vb other i j hh hh hi hj he).2.1
    omega

#axiomsClean state_read_history
#axiomsClean state_read_final
#axiomsClean output_read
#axiomsClean gradient_read_query
#axiomsClean gradient_read_key
#axiomsClean gradient_read_value
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
