import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.RawOutput

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

theorem tileWrite_congr (a : Arithmetic α) (outputFP8 : Bool) (p : Arguments)
    (pb pm pn : Nat) (v w : Values α .real [16,64]) (s : State α)
    (h : ∀ i,outputActive p pm pn i → v i=w i) :
    tileWrite a outputFP8 p pb pm pn v s=tileWrite a outputFP8 p pb pm pn w s := by
  unfold tileWrite
  apply congrArg (fun f : State α → TileIndex [16,64] → State α =>
    (TileShape.allIndices [16,64]).foldl f s)
  funext t i
  by_cases hi : outputActive p pm pn i
  · simp only [if_pos hi,h i hi]
  · simp only [if_neg hi]

/-- Readable source execution to an independently indexed input and scalar
reference, retaining every final output cast and every untouched cell. -/
theorem source_reference {α : Type} [Inhabited α] (a : Arithmetic α) (z : ZeroLaws a)
    (fp8 outputFP8 : Bool) (p : Arguments) (s : State α) (epsilon theta : α)
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (bytes : Int8Memory s) (even : p.N%2=0) :
    ∃ final, run a.model (source fp8 outputFP8 p).surfaceBody s=some final ∧
      final.mem=(tileWrite a outputFP8 p (s.pids 0) (programRow p s) (programCol p s)
        (rawTile a fp8 p s epsilon theta) s).mem := by
  obtain ⟨final,hex,hmem⟩ := sourceExecution a fp8 outputFP8 p s epsilon theta he ht bytes
  refine ⟨final,hex,hmem.trans (congrArg State.mem (tileWrite_congr a outputFP8 p
    (s.pids 0) (programRow p s) (programCol p s) _ _ s ?_))⟩
  intro i hi
  rcases i with ⟨row,lane,⟨⟩⟩
  exact sourceTile_reference a z fp8 p s epsilon theta even row lane hi.1 hi.2

theorem tileWrite_frame (a : Arithmetic α) (outputFP8 : Bool) (p : Arguments)
    (pb pm pn : Nat) (v : Values α .real [16,64]) (s : State α) (r : RegionName) (o : Nat)
    (miss : r≠p.OUT ∨ ∀ i,outputActive p pm pn i → o≠outputOffset p pb pm pn i) :
    (tileWrite a outputFP8 p pb pm pn v s).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases miss with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun i _ hi => ho i hi)

#axiomsClean tileWrite_congr
#axiomsClean source_reference
#axiomsClean tileWrite_frame
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
