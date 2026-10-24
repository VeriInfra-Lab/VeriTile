import bench.tritonbench_g.streamk_matmul.Proofs.GeneratedSource
namespace VeriTile.Bench.TritonBenchG.StreamK

def firstBody : List Command := match originalSource.first[3]? with
 | some (Command.whileLoop _ body)=>body | _=>[]
def partialBody : List Command := match firstBody[10]? with
 | some (Command.branch _ _ no)=>no | _=>[]
def fullKBody : List Command := match originalSource.fullWave[8]? with
 | some (Command.forLoop _ _ _ body)=>body | _=>[]

theorem original_first_while : originalSource.first[3]? = some
 (.whileLoop (.pure (.lt (.get "fw.start_iter" .zero) (.get "fw.last_iter" .zero))) firstBody) := rfl

theorem original_return_valued_cas_while : partialBody[0]? = some
 (.whileLoop (.casNe (.add (.base .locks) (.get "fw.mac_loop.tile_id" .zero))
   (.lit 1) (.lit 1) (.lit 1)) [.pass]) := rfl

theorem original_full_K_range : originalSource.fullWave[8]? = some
 (.forLoop "ft.k" (.lit 0) (.cdiv (.param .k) (.param .bk)) fullKBody) := rfl

theorem original_unconditional_half_store :
 originalSource.fullWave[9]?=some (.half "ft.acc" "ft.acc") ∧
 originalSource.fullWave[13]?=some (.store .mn (.get "ft.C") "ft.acc") := ⟨rfl,rfl⟩

theorem original_dot_separate_add : fullKBody[2]?=some (.mac "ft.acc" "ft.a" "ft.b") := rfl

#axiomsClean original_first_while
#axiomsClean original_return_valued_cas_while
#axiomsClean original_full_K_range
#axiomsClean original_unconditional_half_store
#axiomsClean original_dot_separate_add
end VeriTile.Bench.TritonBenchG.StreamK
