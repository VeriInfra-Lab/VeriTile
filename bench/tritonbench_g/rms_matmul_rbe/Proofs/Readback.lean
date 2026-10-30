import bench.tritonbench_g.rms_matmul_rbe.Proofs.SourceReference

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 10000000

/-- Only active lanes must have unique addresses. Padding lanes may share an
address with a real row when N is smaller than the fixed 64-column tile. -/
theorem maskedScatter_readback_active {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) (i : ι) (hnd : l.Nodup) (member : i∈l) (hi : active i)
    (unique : ∀ j,active j → off i=off j → i=j) :
    (l.foldl (fun t j => if active j then t.write r (off j) (val j) else t) s).mem r (off i)=val i := by
  obtain ⟨before,after,hl⟩ := List.append_of_mem member
  rw [hl,List.nodup_append,List.nodup_cons] at hnd
  obtain ⟨_,⟨hnot,_⟩,_⟩ := hnd
  rw [hl,List.foldl_append,List.foldl_cons,if_pos hi,State.masked_scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj ha equal
    exact hnot ((unique j ha equal) ▸ hj)

theorem outputOffset_active_injective (p : Arguments) (pb pm pn : Nat)
    (denseRow : p.som=p.N) (denseCol : p.son=1)
    (i j : TileIndex [16,64]) (hi : outputActive p pm pn i) (hj : outputActive p pm pn j)
    (same : outputOffset p pb pm pn i=outputOffset p pb pm pn j) : i=j := by
  have eq : (pm*16+i.1.val)*p.N+(pn*64+i.2.1.val)=
      (pm*16+j.1.val)*p.N+(pn*64+j.2.1.val) := by
    simp only [outputOffset,denseRow,denseCol,Nat.mul_one,Nat.add_assoc,
      Nat.add_left_cancel_iff] at same
    exact same
  have hiN : pn*64+i.2.1.val<p.N := hi.2
  have hjN : pn*64+j.2.1.val<p.N := hj.2
  have hn := congrArg (fun x => x%p.N) eq
  simp only [Nat.mul_add_mod',Nat.mod_eq_of_lt hiN,Nat.mod_eq_of_lt hjN] at hn
  have divide (r n : Nat) (h : n<p.N) : (r*p.N+n)/p.N=r := by
    rw [Nat.mul_comm r p.N,Nat.mul_add_div (by omega),Nat.div_eq_of_lt h,Nat.add_zero]
  have hm := congrArg (fun x => x/p.N) eq
  simp only [divide _ _ hiN,divide _ _ hjN] at hm
  apply Prod.ext (Fin.ext (by omega))
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

theorem tileWrite_readback (a : Arithmetic α) (outputFP8 : Bool) (p : Arguments)
    (pb pm pn : Nat) (v : Values α .real [16,64]) (s : State α)
    (denseRow : p.som=p.N) (denseCol : p.son=1)
    (i : TileIndex [16,64]) (hi : outputActive p pm pn i) :
    (tileWrite a outputFP8 p pb pm pn v s).mem p.OUT (outputOffset p pb pm pn i)=
      outputCell a outputFP8 (v i) :=
  maskedScatter_readback_active _ _ _ _ _ _ i (TileShape.allIndices_nodup _)
    (TileShape.mem_allIndices _ i) hi
    (fun j hj h => outputOffset_active_injective p pb pm pn denseRow denseCol i j hi hj h)

theorem tileWrite_int8 {α : Type} [Inhabited α] (a : Arithmetic α) (outputFP8 : Bool)
    (p : Arguments) (pb pm pn : Nat) (v : Values α .real [16,64])
    (s : State α) (h : Int8Memory s) : Int8Memory (tileWrite a outputFP8 p pb pm pn v s) := by
  suffices aux : ∀ (l : List (TileIndex [16,64])) (s : State α),Int8Memory s →
      Int8Memory (l.foldl (fun t i => if outputActive p pm pn i then
        t.write p.OUT (outputOffset p pb pm pn i) (outputCell a outputFP8 (v i)) else t) s) from
    aux _ s h
  intro l
  induction l with
  | nil => intro s hs; exact hs
  | cons i rest ih =>
    intro s hs
    simp only [List.foldl_cons]
    apply ih
    by_cases active : outputActive p pm pn i
    · simp only [if_pos active]
      intro r o
      simp only [State.write]
      split_ifs
      · cases outputFP8 <;> simp [outputCell,Cell.read,defaultValue,Float8E5Bits.isInt8]
      · exact hs r o
    · simpa only [if_neg active] using hs

#axiomsClean maskedScatter_readback_active
#axiomsClean outputOffset_active_injective
#axiomsClean tileWrite_readback
#axiomsClean tileWrite_int8
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
