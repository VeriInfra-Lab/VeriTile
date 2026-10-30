import bench.tritonbench_g.int8_matmul_quantization.Sources.Kernels
import bench.tritonbench_g.int8_matmul_quantization.Proofs.Coverage
import bench.tritonbench_g.int8_matmul_quantization.Proofs.Host
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Source
noncomputable section
def historicalSourceHash : String := "bb3411459f3accd210a86a225c62d979375870933fbbcb88242e3ee44fa6f1c8"
abbrev productionAST := SourceBinding.productionAST
def config0 : Config := ⟨⟨128,256,64,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config1 : Config := ⟨⟨64,256,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config2 : Config := ⟨⟨128,128,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config3 : Config := ⟨⟨128,64,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config4 : Config := ⟨⟨64,128,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config5 : Config := ⟨⟨128,32,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config6 : Config := ⟨⟨64,32,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config7 : Config := ⟨⟨32,64,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config8 : Config := ⟨⟨64,64,32,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config9 : Config := ⟨⟨64,64,64,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config10 : Config := ⟨⟨32,32,128,1,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config11 : Config := ⟨⟨64,64,32,1,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config12 : Config := ⟨⟨64,64,64,1,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config13 : Config := ⟨⟨32,32,128,1,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config14 : Config := ⟨⟨64,64,64,1,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config15 : Config := ⟨⟨64,64,128,1,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config16 : Config := ⟨⟨32,32,256,1,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config17 : Config := ⟨⟨128,256,64,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config18 : Config := ⟨⟨64,256,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config19 : Config := ⟨⟨128,128,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config20 : Config := ⟨⟨128,64,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config21 : Config := ⟨⟨64,128,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config22 : Config := ⟨⟨128,32,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config23 : Config := ⟨⟨64,32,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config24 : Config := ⟨⟨32,64,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config25 : Config := ⟨⟨64,64,32,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config26 : Config := ⟨⟨64,64,64,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config27 : Config := ⟨⟨32,32,128,2,by decide,by decide,by decide,by decide⟩,8,by decide⟩
def config28 : Config := ⟨⟨64,64,32,2,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config29 : Config := ⟨⟨64,64,64,2,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config30 : Config := ⟨⟨32,32,128,2,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config31 : Config := ⟨⟨64,64,64,2,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config32 : Config := ⟨⟨64,64,128,2,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def config33 : Config := ⟨⟨32,32,256,2,by decide,by decide,by decide,by decide⟩,16,by decide⟩
def configs : List Config := [config0,config1,config2,config3,config4,config5,config6,config7,config8,config9,config10,config11,config12,config13,config14,config15,config16,config17,config18,config19,config20,config21,config22,config23,config24,config25,config26,config27,config28,config29,config30,config31,config32,config33]
def warpsStages : List (Nat × Nat) := [(8,3),(4,4),(4,4),(4,4),(4,4),(4,4),(2,5),(2,5),(4,4),(8,3),(4,2),(4,4),(8,3),(4,2),(4,4),(8,3),(4,2),(8,3),(4,4),(4,4),(4,4),(4,4),(4,4),(2,5),(2,5),(4,4),(8,3),(4,2),(4,4),(8,3),(4,2),(4,4),(8,3),(4,2)]
def quantizerWarpsStages : List (Nat × Nat) := [(8,2),(4,2),(2,2),(1,2)]
def Admitted (c : Config) : Prop := c∈configs
theorem configurations_count : configs.length=34 := rfl
theorem quantizer_count : quantizerWarpsStages.length=4 := rfl
theorem split_counts (c : Config) (h : Admitted c) : c.S=1 ∨ c.S=2 := by
  simp only [Admitted,configs,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals decide
theorem group_counts (c : Config) (h : Admitted c) : c.group=8 ∨ c.group=16 := by
  simp only [Admitted,configs,List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals decide
#axiomsClean split_counts
#axiomsClean group_counts
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Source
