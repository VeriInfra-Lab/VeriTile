import bench.repairs.tritonbench_g.triton_argmax.Proofs.GlobalGrid
import bench.repairs.tritonbench_g.triton_argmax.Proofs.Real

/-! Conditional numerical bridge. The conditions are the scalar comparisons
at each reached node, including comparisons of the explicitly cast stage-one
intermediate. No output equality, reduction law, memory law or successful run
is a numerical premise. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.GlobalBridge
open Source Reduction
noncomputable section

structure LocalNumerics {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) : Prop where
  first : ∀ pid,pid < c.blocks → (c.firstTree pid).LocalAgreement ops.gt32 ops.eq32
    orderedGt orderedEq (c.loaded bits ops.negInf32 pid)
    (Math.padded c.size (pid*c.block) c.block logical)
  second : c.secondTree.LocalAgreement ops.gtMid ops.eqMid orderedGt orderedEq
    (stage2Loaded ops c (stage1Value ops c bits)) (Math.midValues c logical)

theorem first_index {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (bits : Nat → Word) (logical : Nat → ℝ) (num : LocalNumerics ops c bits logical)
    (pid : Nat) (active : pid < c.blocks) : stage1Index ops c bits pid=Math.firstIndex c logical pid := by
  have selected := pick_agrees ops.gt32 ops.eq32 orderedGt orderedEq
    (c.loaded bits ops.negInf32 pid) (Math.padded c.size (pid*c.block) c.block logical)
    (c.firstTree pid) (num.first pid active)
  unfold stage1Index GlobalConfig.selected Math.firstIndex Math.blockSelected
  rw [selected]
  omega

theorem global_index {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (bits : Nat → Word) (logical : Nat → ℝ)
    (num : LocalNumerics ops c bits logical) :
    Source.globalIndex ops c bits=Math.globalIndex c logical := by
  have selected := pick_agrees ops.gtMid ops.eqMid orderedGt orderedEq
    (stage2Loaded ops c (stage1Value ops c bits)) (Math.midValues c logical)
    c.secondTree num.second
  have same : stage2Selected ops c (stage1Value ops c bits)=Math.globalSelected c logical := selected
  unfold Source.globalIndex
  rw [same,if_pos (Math.mid_valid c coverage logical)]
  exact first_index ops c bits logical num _ (Math.mid_valid c coverage logical)

theorem reference_index {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (coverage : Math.GlobalCoverage c) (bits : Nat → Word) (logical : Nat → ℝ)
    (num : LocalNumerics ops c bits logical) :
    Source.globalIndex ops c bits=(Math.argmax c.size coverage.sizePositive
      (fun i : Fin c.size => logical i.val)).val := by
  rw [global_index ops c coverage bits logical num]
  have unique := least_maximum_unique (fun i : Fin c.size => logical i.val)
    ⟨Math.globalIndex c logical,Math.global_valid c coverage logical⟩
    (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => logical i.val))
    (Math.global_mathematical c coverage logical)
    (Math.argmax_spec c.size coverage.sizePositive (fun i : Fin c.size => logical i.val))
  exact congrArg Fin.val unique

theorem returned_bytes {M : Type} {d : MidFormat} (mem : Memory.Primitives M)
    (ops : Scalars d) (c : GlobalConfig) (coverage : Math.GlobalCoverage c)
    (p : GlobalMemory.Buffers) (layout : GlobalMemory.Layout d c p) (s : M)
    (initialized : ∀ i,i < c.midBlock → mem.read 8 s (Memory.address p.midIndex 8 i)=0)
    (logical : Nat → ℝ) (num : LocalNumerics ops c (GlobalMemory.input mem p s) logical) :
    mem.read 8 (GlobalMemory.run mem ops c p s) (Memory.address p.output 8 0)=
      BitVec.ofNat 64 (Math.argmax c.size coverage.sizePositive (fun i : Fin c.size => logical i.val)).val := by
  rw [GlobalMemory.global_readback mem ops c p layout s initialized,
    reference_index ops c coverage (GlobalMemory.input mem p s) logical num]

#axiomsClean first_index
#axiomsClean global_index
#axiomsClean reference_index
#axiomsClean returned_bytes
end
end VeriTile.Bench.Repairs.TritonArgmax.GlobalBridge
