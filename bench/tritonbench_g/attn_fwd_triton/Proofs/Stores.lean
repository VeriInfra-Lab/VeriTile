import bench.tritonbench_g.attn_fwd_triton.Proofs.Initial

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private theorem scatter_at {ι : Type} (r : RegionName) (off : ι→Nat) (value : ι→Cell Expr)
    (active : ι→Prop) [DecidablePred active] (xs : List ι) (s : State Expr)
    (i : ι) (hm : i∈xs) (hi : active i)
    (unique : ∀ j, active j → off j=off i → j=i) :
    (xs.foldl (fun t j=>if active j then t.write r (off j) (value j) else t) s).mem r (off i)=value i := by
  induction xs generalizing s with
  | nil=>simp at hm
  | cons j xs ih=>
    by_cases ht : i∈xs
    · exact ih _ ht
    · have hji : j=i := by simpa [ht,eq_comm] using hm
      subst j
      simp only [List.foldl_cons,if_pos hi]
      rw [State.masked_scatter_frame r off value active xs r (off i)
        (Or.inr (by intro j hj ha he; exact ht ((unique j ha he.symm) ▸ hj)))]
      exact State.write_same ..

theorem output_injective_active (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (hn : NonOverlapping p) (i j : TileIndex [128,BD])
    (hi : outputActive p BD idx.1.val i=Bool.true)
    (hj : outputActive p BD idx.1.val j=Bool.true)
    (he : outputAddress p BD idx.1.val idx.2.val j=outputAddress p BD idx.1.val idx.2.val i) : j=i := by
  have h := hn (show logicalAddress p (coordinate p BD idx j hj)=
    logicalAddress p (coordinate p BD idx i hi) from he)
  have hr := congrArg (fun c : Coordinate p=>c.2.2.1.val) h
  have hd := congrArg (fun c : Coordinate p=>c.2.2.2.val) h
  change idx.1.val*128+j.1.val=idx.1.val*128+i.1.val at hr
  change j.2.1.val=i.2.1.val at hd
  exact Prod.ext (Fin.ext (by omega)) (Prod.ext (Fin.ext hd) (Subsingleton.elim _ _))

theorem bound_ready (f : Fixed BD) (c : Carry BD) (s : State Expr) (hb : Bound f c s) : Ready BD s := by
  rcases hb with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14⟩
  exact ⟨⟨_,h0⟩,⟨_,h1⟩,⟨_,h2⟩,⟨_,h3⟩,⟨_,h4⟩,⟨_,h5⟩,⟨_,h6⟩,⟨_,h7⟩,
    ⟨_,h8⟩,⟨_,h9⟩,⟨_,h10⟩,⟨_,h11⟩,⟨_,h12⟩,⟨_,h13⟩,⟨_,h14⟩⟩

theorem tail_read (p : Config) (BD : Nat) (idx : ProgramIndex p)
    (memory : RegionName→Nat→Cell Expr) (c : Carry BD) (s t : State Expr)
    (hb : Bound (fixed p BD memory idx.1.val idx.2.val) c s)
    (hn : NonOverlapping p) (ht : run (engine backend) (tail p BD Bool.false) s=some t)
    (i : TileIndex [128,BD]) (hi : outputActive p BD idx.1.val i=Bool.true) :
    t.mem p.Out (outputAddress p BD idx.1.val idx.2.val i)=Cell.mk .bf16 (output c i) := by
  rcases hb with ⟨h0,_,_,_,h4,_,_,_,_,h9,_,_,_,h13,h14⟩
  simp only [fixed] at h0 h4 h9
  simp only [tail,source,Bool.false_eq_true,↓reduceIte,AttnFwdTriton.TypedSource.repaired,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero] at ht
  simp [run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,natLt,numericLt,
    Cell.read,ofFloat,toFloat,Region.cast,State.setReg,h0,h4,h9,h13,h14,
    engine,Algebra.withDefaultPrecision,resolvePrecision,backend,Graph.cast,nativeDType,consume,bop] at ht
  subst t
  apply scatter_at
  · exact TileShape.mem_allIndices _ i
  · simpa [outputActive] using hi
  · intro j hj he
    exact output_injective_active p BD idx hn i j hi (by simpa [outputActive] using hj) he

#axiomsClean output_injective_active
#axiomsClean tail_read
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked.Model
