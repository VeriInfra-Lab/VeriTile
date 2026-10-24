import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.TypedExecution

namespace VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open TypedExecution
noncomputable section
open scoped Classical
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def Frame {α : Type} (f : Flags) (r : Regions) (stride : Strides)
    (M N row : Nat) (s t : State α) : Prop :=
  ∀ region offset, ¬ Contract.written f r stride M N row region offset →
    t.mem region offset = s.mem region offset

theorem Frame.trans {α : Type} {s t u : State α} (h₁ : Frame f r stride M N row s t)
    (h₂ : Frame f r stride M N row t u) : Frame (α:=α) f r stride M N row s u :=
  fun q k hk => (h₂ q k hk).trans (h₁ q k hk)

/-- Exact tagged-cell frame, including holes and masked tail lanes. -/
theorem vector_frame {α : Type} (dest region : RegionName) (base N offset block : Nat)
    (val : TileIndex [block] → Cell α) (s : State α)
    (h : ¬ (region=dest ∧ Contract.inRow base N offset)) :
    ((TileShape.allIndices [block]).foldl (fun t j => if j.1.val<N then
      t.write dest (base+j.1.val) (val j) else t) s).mem region offset = s.mem region offset := by
  apply State.masked_scatter_frame
  by_cases hr : region=dest
  · right
    intro j _ hj he
    exact h ⟨hr,he ▸ Contract.inRow_of_col hj⟩
  · exact Or.inl hr

theorem scalar_frame {α : Type} (dest region : RegionName) (row offset : Nat)
    (val : TileIndex [] → Cell α) (s : State α)
    (h : ¬ (region=dest ∧ offset=row)) :
    ((TileShape.allIndices []).foldl (fun t j =>
      t.write dest row (val j)) s).mem region offset = s.mem region offset := by
  apply State.scatter_frame
  by_cases hr : region=dest
  · exact Or.inr (fun _ _ he => h ⟨hr,he⟩)
  · exact Or.inl hr

theorem prefix_frame {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block)
    (s t : State α) (ht : run (engine A) (stagePrefix f r stride M N eps p block hb) s=some t) :
    t.mem=s.mem := by
  rcases f with ⟨rms,res,resout,bias,drop,mask,scale,x1,w1,b1⟩
  cases res <;> cases resout <;> cases x1 <;> cases w1
  all_goals simp [stagePrefix,repaired,original,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,bop,natLt,
    numericLt,Cell.read,ofFloat,toFloat,Region.cast,Random.i32ArangeSize,
    State.setReg] at ht
  all_goals subst t; rfl

theorem firstStream_frame {α : Type} [Inhabited α] (A : Algebra α) (cmp : Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) (row : Nat)
    (s t : State α) (hs : Ready f r stride block row s)
    (ht : run (engine A) (firstStream f r stride M N eps p block hb) s=some t) :
    Frame f r stride M N row s t := by
  rcases hs with ⟨hrow,hcols,hy,hx1,hres,hresout,hy1,x,hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid ((s.mem r.seeds row).read .int) hb)
  have ho : ∀ j : Fin block, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j,PUnit.unit)
  have hu := hv.1
  simp at hu
  intro region offset hw
  have hm : f.dropout=Bool.true → f.dropoutMask=Bool.true →
      ¬ (region=r.dropoutMask ∧ Contract.inRow (row*N) N offset) := by
    intro hd hm h; exact hw (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨hd,hm,h⟩))))))
  cases hsc : f.rowscale <;> cases hd : f.dropout <;> cases hmask : f.dropoutMask
  all_goals simp [firstStream,repaired,original,ComputeKernel.surfaceBody,
    run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralMod,FP.Structural.bop,natLt,
    numericLt,cmp.available,seed_valid,Region.cast,Random.i32ArangeSize,
    hrow,hcols,hy,hx1,hres,hresout,hy1,hx,hu,ho,hsc,hd,hmask] at ht
  all_goals subst t
  all_goals try simp only [State.setReg_mem]
  all_goals first | rfl | exact vector_frame _ _ _ _ _ _ _ _ (hm hd hmask)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.TypedSource
