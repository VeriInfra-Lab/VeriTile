import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSourceOps

/-! Compose the actual eight source statements. Pointer safety is established
by each executed source constructor, rather than assumed of initial registers.
Only active logical elements require finite allocation bounds. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]
variable (A : FlatAlloc) (M : Algebra α)
variable (background : RegionName → Nat → Cell α)
variable (flatUndef : (d : TileDType) → RegionName → Nat → Value α d)

theorem source_loop_tile_same (s : State α) :
    OpSame A M background flatUndef s none sourceLoopTile := by
  apply op_add A M background flatUndef s none
  · exact op_ref_data A M background flatUndef s none .nat [] "pid" (by decide) (by decide)
  · apply op_mul A M background flatUndef s none
    · exact op_ref_data A M background flatUndef s none .nat [] "j" (by decide) (by decide)
    · exact op_ref_data A M background flatUndef s none .nat [] "num_ctas" (by decide) (by decide)

theorem source_body_transport (hd : A.Disjoint) (p : Parametric.Parameters)
    (wide : Bool) (tile : Op .nat []) (s : State α)
    (htile : OpSame A M background flatUndef s none tile)
    (hu : UndefCompatible A flatUndef s .real)
    (hinput : p.input ∈ A.regions) (houtput : p.output ∈ A.regions)
    (hin : ∀ x, x < p.size → x * p.inputStride < A.extent p.input)
    (hout : ∀ x, x < p.size → x * p.outputStride < A.extent p.output) :
    RunTransport A M background flatUndef (sourceBody p wide tile) s := by
  unfold sourceBody
  refine run_cons_transport A M background flatUndef _ _ s ?_ ?_
  · exact assign_alg_data_transport A M background flatUndef .nat [] "tile_id" tile s
      (by decide) (by decide) htile
  intro s1 h1
  have hu1 := assign_compatible A M flatUndef _ _ _ _ s s1 h1 hu
  refine run_cons_transport A M background flatUndef _ _ s1 ?_ ?_
  · exact assign_alg_data_transport A M background flatUndef .nat [] "tile_id0" _ s1
      (by decide) (by decide)
      (op_ref_data A M background flatUndef s1 none .nat [] "tile_id" (by decide) (by decide))
  intro s2 h2
  have hu2 := assign_compatible A M flatUndef _ _ _ _ s1 s2 h2 hu1
  refine run_cons_transport A M background flatUndef _ _ s2 ?_ ?_
  · apply assign_alg_data_transport A M background flatUndef .nat [] "offset0" _ s2
      (by decide) (by decide)
    apply op_mul A M background flatUndef s2 none
    · exact op_ref_data A M background flatUndef s2 none .nat [] "tile_id0" (by decide) (by decide)
    · exact op_constNat A M background flatUndef s2 none p.block
  intro s3 h3
  have hu3 := assign_compatible A M flatUndef _ _ _ _ s2 s3 h3 hu2
  refine run_cons_transport A M background flatUndef _ _ s3 ?_ ?_
  · exact assign_alg_transport A M background flatUndef .blockPtr [p.block] "in0_bptr" _ s3
      (source_ptr_commutes A M background flatUndef p p.input p.inputStride s3)
  intro s4 h4
  have hu4 := assign_compatible A M flatUndef _ _ _ _ s3 s4 h4 hu3
  have hsafe := source_ptr_safe_after A M p p.input p.inputStride "in0_bptr" s3 s4 hinput hin h4
  refine run_cons_transport A M background flatUndef _ _ s4 ?_ ?_
  · apply assign_transport A M background flatUndef .real [p.block] "in0" _ s4
    exact (exprSame_iff_commutes A M background flatUndef s4 _ (by decide) (by decide)).mp
      (source_load_same A M background flatUndef hd p wide s4 hsafe hu4)
  intro s5 _h5
  refine run_cons_transport A M background flatUndef _ _ s5 ?_ ?_
  · apply assign_transport A M background flatUndef .bool [p.block] "out0" _ s5
    exact (exprSame_iff_commutes A M background flatUndef s5 _ (by decide) (by decide)).mp
      (source_classify_same A M background flatUndef p wide s5)
  intro s6 _h6
  refine run_cons_transport A M background flatUndef _ _ s6 ?_ ?_
  · exact assign_alg_transport A M background flatUndef .blockPtr [p.block] "out0_bptr" _ s6
      (source_ptr_commutes A M background flatUndef p p.output p.outputStride s6)
  intro s7 h7
  have hsafeOut := source_ptr_safe_after A M p p.output p.outputStride "out0_bptr" s6 s7 houtput hout h7
  refine run_cons_transport A M background flatUndef _ _ s7 ?_ ?_
  · intro final hf
    have hvalue := expr_alg A M background flatUndef s7 (.ref .bool [p.block] "out0")
      (op_ref_data A M background flatUndef s7 none .bool [p.block] "out0" (by decide) (by decide))
    have hstore := step_store_bool_none A M background flatUndef s7 hd
      (.ref .blockPtr [p.block] "out0_bptr") [0] (.alg (.ref .bool [p.block] "out0"))
      hvalue (op_ref_commutes A M background flatUndef s7 none .blockPtr [p.block] "out0_bptr")
      (fun ps hp => hsafeOut ps hp)
    rw [hf, Option.map_some] at hstore
    exact hstore
  intro s8 _h8
  exact run_nil_transport A M background flatUndef s8

theorem blockStoreBool_undef {sh : TileShape} (checked : List Nat)
    (ps : Values α .blockPtr sh) (values : Values α .bool sh) (s : State α) :
    (blockStoreBool checked ps values s).undef = s.undef := by
  have h (lanes : List (TileIndex sh)) (t : State α) :
      (lanes.foldl (fun u i =>
        if (ps i).inBounds (TileShape.indexToList sh i) checked then
          u.write (ps i).region ((ps i).address (TileShape.indexToList sh i)) (.mk .bool (values i))
        else u) t).undef = t.undef := by
    induction lanes generalizing t with
    | nil => rfl
    | cons i rest ih =>
        simp only [List.foldl_cons]
        split <;> rw [ih] <;> rfl
  exact h _ s

theorem step_store_undef {sh : TileShape} (ptr : Op .blockPtr sh) (checked : List Nat)
    (value : ComputeExpr .bool sh) (s t : State α)
    (ht : FP.Structural.step M (.store .bool sh (.blockPtr ptr checked) value .none) s = some t) :
    t.undef = s.undef := by
  rw [FP.Structural.step] at ht
  cases hv : evalExpr M value s with
  | none => simp only [hv, bind, Option.bind_none] at ht; cases ht
  | some values =>
      simp only [hv, bind, Option.bind_some] at ht
      rw [store_blockPtr_bool_none] at ht
      cases hp : evalOp M none ptr s with
      | none => simp only [hp, Option.map_none] at ht; cases ht
      | some ps =>
          simp only [hp, Option.map_some, Option.some.injEq] at ht
          subst t
          exact blockStoreBool_undef checked ps values s

theorem run_undef (body : List ComputeStmt)
    (hsteps : ∀ st ∈ body, ∀ s t : State α,
      FP.Structural.step M st s = some t → t.undef = s.undef)
    (s t : State α) (ht : FP.Structural.run M body s = some t) : t.undef = s.undef := by
  induction body generalizing s with
  | nil =>
      rw [FP.Structural.run] at ht
      cases Option.some.inj ht
      rfl
  | cons st rest ih =>
      rw [FP.Structural.run] at ht
      cases he : FP.Structural.step M st s with
      | none => simp only [he, bind, Option.bind_none] at ht; cases ht
      | some u =>
          simp only [he, bind, Option.bind_some] at ht
          exact (ih (fun q hq => hsteps q (List.mem_cons_of_mem st hq)) u ht).trans
            (hsteps st (List.mem_cons_self ..) s u he)

theorem source_body_undef (p : Parametric.Parameters) (wide : Bool) (tile : Op .nat [])
    (s t : State α) (ht : FP.Structural.run M (sourceBody p wide tile) s = some t) :
    t.undef = s.undef := by
  apply run_undef M (sourceBody p wide tile) ?_ s t ht
  intro st hst u v hv
  simp only [sourceBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first
    | exact assign_undef M _ _ _ _ u v hv
    | exact step_store_undef M _ _ _ u v hv

#axiomsClean source_loop_tile_same
#axiomsClean source_body_transport
#axiomsClean blockStoreBool_undef
#axiomsClean step_store_undef
#axiomsClean run_undef
#axiomsClean source_body_undef
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedFlat
