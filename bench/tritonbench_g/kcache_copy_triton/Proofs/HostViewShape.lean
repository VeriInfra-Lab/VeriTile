import bench.tritonbench_g.kcache_copy_triton.Proofs.HostPrepare

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

theorem consume_length (newRev : List Nat) (target base product : Nat) :
    (consume newRev target base product).1.length +
      (consume newRev target base product).2.2.length = newRev.length := by
  induction newRev generalizing product with
  | nil => rfl
  | cons n ns ih =>
    simp only [consume]
    split
    · simpa only [List.length_cons] using congrArg Nat.succ (ih (product*n))
    · simp

theorem chunks_length (oldRev : List (Nat × Nat)) (newRev : List Nat)
    (base chunk : Nat) (strides : List Nat)
    (success : inferChunks oldRev newRev base chunk = some strides) :
    strides.length = newRev.length := by
  induction oldRev generalizing newRev base chunk strides with
  | nil =>
    simp only [inferChunks] at success
    split at success
    · simp_all
    · contradiction
  | cons entry oldRev ih =>
    rcases entry with ⟨n,s⟩
    have mapped (nextBase : Nat)
        (hs : (inferChunks oldRev (consume newRev (chunk*n) base 1).1 nextBase 1).map
          ((consume newRev (chunk*n) base 1).2.2 ++ ·) = some strides) :
        strides.length = newRev.length := by
      obtain ⟨next,hn,he⟩ := Option.map_eq_some_iff.mp hs
      subst strides
      have hl := ih _ _ _ next hn
      simp only [List.length_append,hl]
      have hc := consume_length newRev (chunk*n) base 1
      omega
    rw [inferChunks.eq_def] at success
    cases oldRev with
    | nil =>
      dsimp only at success
      split_ifs at success
      all_goals first | exact mapped _ success | exact ih _ _ _ strides success | contradiction
    | cons entry rest =>
      rcases entry with ⟨previousSize,previousStride⟩
      dsimp only at success
      split_ifs at success
      all_goals first | exact mapped _ success | exact ih _ _ _ strides success | contradiction

theorem rank4_view_length (a b h d sa sb sh sd : Nat) (strides : List Nat)
    (success : inferViewStrides [a,b,h,d] [sa,sb,sh,sd] [a*b,h,d] = some strides) :
    strides.length = 3 := by
  unfold inferViewStrides at success
  simp only [List.cons_ne_nil,↓reduceIte] at success
  split at success
  · have different : [a,b,h,d] ≠ [a*b,h,d] := by
      intro eq
      have := congrArg List.length eq
      simp only [List.length_cons,List.length_nil] at this
      omega
    simp only [different,↓reduceIte] at success
    cases success
    rfl
  · obtain ⟨rev,hr,he⟩ := Option.map_eq_some_iff.mp success
    subst strides
    have hl := chunks_length _ _ _ _ rev hr
    simpa only [List.length_reverse,List.length_cons,List.length_nil] using hl

#axiomsClean consume_length
#axiomsClean chunks_length
#axiomsClean rank4_view_length
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.View
