import bench.tritonbench_g.llama_ff_triton.Proofs.Core
namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxRecDepth 20000
set_option maxHeartbeats 8000000
/-- One source K64 block contains four accumulator-fed m16n8k16 operations. -/
def orderedK16 : DotPlan where
  instructions := (List.finRange 4).map (fun q => fun l : Fin 16 =>
    (⟨q.val*16+l.val,by have hq:=q.isLt; have hl:=l.isLt; omega⟩ : Fin 64))
  complete := by decide
/-- SM90 selected profile: warp butterflies 16,8,4,2,1, followed by
the inter-warp addition. The full finite tree is explicit. -/
def butterfly64 : Reduction where
  tree := (.node (.node (.node (.node (.node (.node (.leaf ⟨0,by decide⟩) (.leaf ⟨16,by decide⟩)) (.node (.leaf ⟨8,by decide⟩) (.leaf ⟨24,by decide⟩))) (.node (.node (.leaf ⟨4,by decide⟩) (.leaf ⟨20,by decide⟩)) (.node (.leaf ⟨12,by decide⟩) (.leaf ⟨28,by decide⟩)))) (.node (.node (.node (.leaf ⟨2,by decide⟩) (.leaf ⟨18,by decide⟩)) (.node (.leaf ⟨10,by decide⟩) (.leaf ⟨26,by decide⟩))) (.node (.node (.leaf ⟨6,by decide⟩) (.leaf ⟨22,by decide⟩)) (.node (.leaf ⟨14,by decide⟩) (.leaf ⟨30,by decide⟩))))) (.node (.node (.node (.node (.leaf ⟨1,by decide⟩) (.leaf ⟨17,by decide⟩)) (.node (.leaf ⟨9,by decide⟩) (.leaf ⟨25,by decide⟩))) (.node (.node (.leaf ⟨5,by decide⟩) (.leaf ⟨21,by decide⟩)) (.node (.leaf ⟨13,by decide⟩) (.leaf ⟨29,by decide⟩)))) (.node (.node (.node (.leaf ⟨3,by decide⟩) (.leaf ⟨19,by decide⟩)) (.node (.leaf ⟨11,by decide⟩) (.leaf ⟨27,by decide⟩))) (.node (.node (.leaf ⟨7,by decide⟩) (.leaf ⟨23,by decide⟩)) (.node (.leaf ⟨15,by decide⟩) (.leaf ⟨31,by decide⟩)))))) (.node (.node (.node (.node (.node (.leaf ⟨32,by decide⟩) (.leaf ⟨48,by decide⟩)) (.node (.leaf ⟨40,by decide⟩) (.leaf ⟨56,by decide⟩))) (.node (.node (.leaf ⟨36,by decide⟩) (.leaf ⟨52,by decide⟩)) (.node (.leaf ⟨44,by decide⟩) (.leaf ⟨60,by decide⟩)))) (.node (.node (.node (.leaf ⟨34,by decide⟩) (.leaf ⟨50,by decide⟩)) (.node (.leaf ⟨42,by decide⟩) (.leaf ⟨58,by decide⟩))) (.node (.node (.leaf ⟨38,by decide⟩) (.leaf ⟨54,by decide⟩)) (.node (.leaf ⟨46,by decide⟩) (.leaf ⟨62,by decide⟩))))) (.node (.node (.node (.node (.leaf ⟨33,by decide⟩) (.leaf ⟨49,by decide⟩)) (.node (.leaf ⟨41,by decide⟩) (.leaf ⟨57,by decide⟩))) (.node (.node (.leaf ⟨37,by decide⟩) (.leaf ⟨53,by decide⟩)) (.node (.leaf ⟨45,by decide⟩) (.leaf ⟨61,by decide⟩)))) (.node (.node (.node (.leaf ⟨35,by decide⟩) (.leaf ⟨51,by decide⟩)) (.node (.leaf ⟨43,by decide⟩) (.leaf ⟨59,by decide⟩))) (.node (.node (.leaf ⟨39,by decide⟩) (.leaf ⟨55,by decide⟩)) (.node (.leaf ⟨47,by decide⟩) (.leaf ⟨63,by decide⟩)))))))
  complete := by decide
example : orderedK16.instructions.length=4 := by decide
example : butterfly64.tree.leaves.length=64 := by decide
#axiomsClean orderedK16
#axiomsClean butterfly64
end
end VeriTile.Bench.Repairs.LlamaFF
