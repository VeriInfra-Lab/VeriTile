import bench.tritonbench_g.mixed_sparse_attention.Proofs.FusedContext

namespace VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
open VeriTile Triton FP.Structural
open Whole002 Source
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem split_code (xs : List ComputeStmt) (n k : Nat) :
    xs = xs.take n ++ (xs.drop n).take k ++ xs.drop (n+k) := by
  simp only [← List.drop_drop]
  rw [List.append_assoc,List.take_append_drop,List.take_append_drop]

theorem column_prefix (c : Config) : (columnBody c).take 5=(column c).take 5 := by
  cases hs : c.storage <;>
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,Source.kernel,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,hs]

theorem column_suffix (c : Config) : (columnBody c).drop 7=(column c).drop 6 := by
  cases hs : c.storage <;>
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,Source.kernel,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,hs]

theorem column_equal (M : Algebra α) (c : Config) : CodeEq M (columnBody c) (column c) := by
  have pre:=CodeEq.refl M ((columnBody c).take 5)
  rw [column_prefix] at pre
  have mid : CodeEq M ((columnBody c).drop 5 |>.take 2) ((column c).drop 5 |>.take 1) :=
    ⟨local_run M c,local_safe M c⟩
  have post:=CodeEq.refl M ((columnBody c).drop 7)
  rw [column_suffix] at post
  rw [split_code (columnBody c) 5 2,split_code (column c) 5 1]
  exact (CodeEq.append (CodeEq.append (by simpa only [column_prefix] using pre) mid)
    (by change CodeEq M ((columnBody c).drop 7) ((column c).drop 6); rw [column_suffix]; exact CodeEq.refl M _))

def columnStmt (c : Config) : ComputeStmt :=
  .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "max_num_cols") (.constNat c.BN) (column c)

theorem active_prefix (c : Config) : (activeBody c).take 23=(active c).take 23 := by
  cases hs : c.storage <;>
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,Source.kernel,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,hs,columnStatement,columnStmt]

theorem active_suffix (c : Config) : (activeBody c).drop 24=(active c).drop 24 := by
  cases hs : c.storage <;>
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,Source.kernel,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,hs,columnStatement,columnStmt]

theorem active_middle (c : Config) :
    ((activeBody c).drop 23).take 1=[columnStatement c] ∧
    ((active c).drop 23).take 1=[columnStmt c] := by
  cases hs : c.storage <;>
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,Source.kernel,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,hs,columnStatement,columnStmt]

theorem active_equal (M : Algebra α) (c : Config) : CodeEq M (activeBody c) (active c) := by
  rw [split_code (activeBody c) 23 1,split_code (active c) 23 1]
  rw [(active_middle c).1,(active_middle c).2]
  apply CodeEq.append
  · apply CodeEq.append
    · rw [active_prefix]; exact CodeEq.refl M _
    · exact (dynamic_context (column_equal M c) _ _ _ _).singleton
  · rw [active_suffix]; exact CodeEq.refl M _

def guard (c : Config) : Op .bool [] :=
  let code : List ComputeStmt := (Source.kernel c).surfaceBody
  match code[3]? with
  | some (ComputeStmt.ifThen cond _)=>cond
  | _=>.constBool Bool.false

theorem source_shape (c : Config) :
    (Source.kernel c).surfaceBody=(parsedKernel c).surfaceBody.take 3 ++ [.ifThen (guard c) (activeBody c)] ∧
    (kernel c).surfaceBody=(parsedKernel c).surfaceBody.take 3 ++ [.ifThen (guard c) (active c)] := by
  cases hs : c.storage <;>
    simp [column,columnBody,active,activeBody,parsedActiveBody,parsedKernel,kernel,Source.kernel,
      TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.FP16.fusedMask,TypedSource.BF16.fusedMask,
      ComputeKernel.surfaceBody,hs,columnStatement,columnStmt,guard]

theorem kernel_equal (M : Algebra α) (c : Config) :
    CodeEq M (Source.kernel c).surfaceBody (kernel c).surfaceBody := by
  rw [(source_shape c).1,(source_shape c).2]
  exact (CodeEq.refl M _).append (conditional_context (active_equal M c) (guard c)).singleton

theorem execution_equal (M : Algebra α) (c : Config) (s : State α) :
    FP.Structural.exec M (Source.kernel c) s=FP.Structural.exec M (kernel c) s := by
  cases h : Source.kernel c
  cases hc : kernel c
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (kernel_equal M c).run s

theorem safety_equal (M : Algebra α) (c : Config) (cap : RegionName → Nat) (s : State α) :
    TraceSafe M cap (Source.kernel c) s ↔ TraceSafe M cap (kernel c) s :=
  (kernel_equal M c).safe cap s

#axiomsClean execution_equal
#axiomsClean safety_equal
end
end VeriTile.Bench.Repairs.MixedSparseAttention.FusedMask
