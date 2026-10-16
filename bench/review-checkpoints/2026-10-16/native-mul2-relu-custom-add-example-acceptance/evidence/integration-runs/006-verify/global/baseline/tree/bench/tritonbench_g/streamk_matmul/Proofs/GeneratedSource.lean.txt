-- GENERATED from all five unchanged pinned JIT ASTs; helpers are inlined.
-- Python parsing/typed lowering is an explicit unverified frontend boundary.
import bench.tritonbench_g.streamk_matmul.Proofs.Control
namespace VeriTile.Bench.TritonBenchG.StreamK

def originalSource : Source where
 first := [
  .integer "fw.pid" .scalar .pid,
  .integer "fw.start_iter" .scalar (.add (.mul (.get "fw.pid" .zero) (.param .fullIters)) (.min (.get "fw.pid" .zero) (.param .partialIters))),
  .integer "fw.last_iter" .scalar (.add (.mul (.add (.get "fw.pid" .zero) (.lit 1)) (.param .fullIters)) (.min (.add (.get "fw.pid" .zero) (.lit 1)) (.param .partialIters))),
  .whileLoop (.pure (.lt (.get "fw.start_iter" .zero) (.get "fw.last_iter" .zero))) [
  .integer "fw.end_iter" .scalar (.min (.add (.get "fw.start_iter" .zero) (.sub (.param .itersPerTile) (.srem (.get "fw.start_iter" .zero) (.param .itersPerTile)))) (.get "fw.last_iter" .zero)),
  .integer "fw.mac_loop.tile_id" .scalar (.sdiv (.get "fw.start_iter" .zero) (.param .itersPerTile)),
  .branch (.gt (.param .groupM) (.lit 0)) [
  .integer "fw.mac_loop.swizzle_tile.grid_m" .scalar (.cdiv (.param .m) (.param .bm)),
  .integer "fw.mac_loop.swizzle_tile.grid_n" .scalar (.cdiv (.param .n) (.param .bn)),
  .integer "fw.mac_loop.swizzle_tile.width" .scalar (.mul (.param .groupM) (.get "fw.mac_loop.swizzle_tile.grid_n" .zero)),
  .integer "fw.mac_loop.swizzle_tile.group_id" .scalar (.sdiv (.get "fw.mac_loop.tile_id" .zero) (.get "fw.mac_loop.swizzle_tile.width" .zero)),
  .integer "fw.mac_loop.swizzle_tile.group_size" .scalar (.min (.sub (.get "fw.mac_loop.swizzle_tile.grid_m" .zero) (.mul (.get "fw.mac_loop.swizzle_tile.group_id" .zero) (.param .groupM))) (.param .groupM)),
  .integer "fw.mac_loop.swizzle_tile.pid_m" .scalar (.add (.mul (.get "fw.mac_loop.swizzle_tile.group_id" .zero) (.param .groupM)) (.srem (.get "fw.mac_loop.tile_id" .zero) (.get "fw.mac_loop.swizzle_tile.group_size" .zero))),
  .integer "fw.mac_loop.swizzle_tile.pid_n" .scalar (.sdiv (.srem (.get "fw.mac_loop.tile_id" .zero) (.get "fw.mac_loop.swizzle_tile.width" .zero)) (.get "fw.mac_loop.swizzle_tile.group_size" .zero)),
  .integer "fw.mac_loop.pid_m" .scalar (.get "fw.mac_loop.swizzle_tile.pid_m" .zero),
  .integer "fw.mac_loop.pid_n" .scalar (.get "fw.mac_loop.swizzle_tile.pid_n" .zero)
] [
  .integer "fw.mac_loop.linear_tile.pid_m" .scalar (.sdiv (.get "fw.mac_loop.tile_id" .zero) (.cdiv (.param .n) (.param .bn))),
  .integer "fw.mac_loop.linear_tile.pid_n" .scalar (.srem (.get "fw.mac_loop.tile_id" .zero) (.cdiv (.param .n) (.param .bn))),
  .integer "fw.mac_loop.pid_m" .scalar (.get "fw.mac_loop.linear_tile.pid_m" .zero),
  .integer "fw.mac_loop.pid_n" .scalar (.get "fw.mac_loop.linear_tile.pid_n" .zero)
],
  .integer "fw.mac_loop.rm" .m (.add (.mul (.get "fw.mac_loop.pid_m" .zero) (.param .bm)) (.axis .index)),
  .integer "fw.mac_loop.rn" .n (.add (.mul (.get "fw.mac_loop.pid_n" .zero) (.param .bn)) (.axis .index)),
  .integer "fw.mac_loop.rk" .k (.axis .index),
  .pointer "fw.mac_loop.A" .mk (.add (.add (.base .a) (.add (.mul (.get "fw.mac_loop.rm" .row) (.param .strideAM)) (.mul (.get "fw.mac_loop.rk" .inner) (.param .strideAK)))) (.mul (.mul (.param .bk) (.param .strideAK)) (.srem (.get "fw.start_iter" .zero) (.param .itersPerTile)))),
  .pointer "fw.mac_loop.B" .kn (.add (.add (.base .b) (.add (.mul (.get "fw.mac_loop.rk" .inner) (.param .strideBK)) (.mul (.get "fw.mac_loop.rn" .col) (.param .strideBN)))) (.mul (.mul (.param .bk) (.param .strideBK)) (.srem (.get "fw.start_iter" .zero) (.param .itersPerTile)))),
  .zero "fw.mac_loop.acc" .mn,
  .forLoop "fw.mac_loop.current_iter" (.get "fw.start_iter" .zero) (.get "fw.end_iter" .zero) [
  .load "fw.mac_loop.a" .mk (.get "fw.mac_loop.A"),
  .load "fw.mac_loop.b" .kn (.get "fw.mac_loop.B"),
  .mac "fw.mac_loop.acc" "fw.mac_loop.a" "fw.mac_loop.b",
  .pointer "fw.mac_loop.A" .mk (.add (.get "fw.mac_loop.A") (.mul (.param .bk) (.param .strideAK))),
  .pointer "fw.mac_loop.B" .kn (.add (.get "fw.mac_loop.B") (.mul (.param .bk) (.param .strideBK)))
],
  .branch (.eq (.srem (.get "fw.end_iter" .zero) (.param .itersPerTile)) (.lit 0)) [
  .pointer "fw.mac_loop.C_" .mn (.add (.base .c) (.add (.mul (.get "fw.mac_loop.rm" .row) (.param .strideCM)) (.mul (.get "fw.mac_loop.rn" .col) (.param .strideCN)))),
  .store .mn (.get "fw.mac_loop.C_") "fw.mac_loop.acc",
  .branch (.ne (.srem (.get "fw.start_iter" .zero) (.param .itersPerTile)) (.lit 0)) [
  .exchange (.add (.base .locks) (.get "fw.mac_loop.tile_id" .zero)) (.lit 1)
] [
  
]
] [
  .whileLoop (.casNe (.add (.base .locks) (.get "fw.mac_loop.tile_id" .zero)) (.lit 1) (.lit 1) (.lit 1)) [
  .pass
],
  .pointer "fw.mac_loop.C_" .mn (.add (.base .c) (.add (.mul (.get "fw.mac_loop.rm" .row) (.param .strideCM)) (.mul (.get "fw.mac_loop.rn" .col) (.param .strideCN)))),
  .atomicAdd .mn (.get "fw.mac_loop.C_") "fw.mac_loop.acc"
],
  .integer "fw.start_iter" .scalar (.get "fw.end_iter" .zero)
]
]
 fullWave := [
  .integer "ft.tile_id" .scalar (.add .pid (.param .streamTiles)),
  .branch (.gt (.param .groupM) (.lit 0)) [
  .integer "ft.swizzle_tile.grid_m" .scalar (.cdiv (.param .m) (.param .bm)),
  .integer "ft.swizzle_tile.grid_n" .scalar (.cdiv (.param .n) (.param .bn)),
  .integer "ft.swizzle_tile.width" .scalar (.mul (.param .groupM) (.get "ft.swizzle_tile.grid_n" .zero)),
  .integer "ft.swizzle_tile.group_id" .scalar (.sdiv (.get "ft.tile_id" .zero) (.get "ft.swizzle_tile.width" .zero)),
  .integer "ft.swizzle_tile.group_size" .scalar (.min (.sub (.get "ft.swizzle_tile.grid_m" .zero) (.mul (.get "ft.swizzle_tile.group_id" .zero) (.param .groupM))) (.param .groupM)),
  .integer "ft.swizzle_tile.pid_m" .scalar (.add (.mul (.get "ft.swizzle_tile.group_id" .zero) (.param .groupM)) (.srem (.get "ft.tile_id" .zero) (.get "ft.swizzle_tile.group_size" .zero))),
  .integer "ft.swizzle_tile.pid_n" .scalar (.sdiv (.srem (.get "ft.tile_id" .zero) (.get "ft.swizzle_tile.width" .zero)) (.get "ft.swizzle_tile.group_size" .zero)),
  .integer "ft.pid_m" .scalar (.get "ft.swizzle_tile.pid_m" .zero),
  .integer "ft.pid_n" .scalar (.get "ft.swizzle_tile.pid_n" .zero)
] [
  .integer "ft.linear_tile.pid_m" .scalar (.sdiv (.get "ft.tile_id" .zero) (.cdiv (.param .n) (.param .bn))),
  .integer "ft.linear_tile.pid_n" .scalar (.srem (.get "ft.tile_id" .zero) (.cdiv (.param .n) (.param .bn))),
  .integer "ft.pid_m" .scalar (.get "ft.linear_tile.pid_m" .zero),
  .integer "ft.pid_n" .scalar (.get "ft.linear_tile.pid_n" .zero)
],
  .integer "ft.rm" .m (.add (.mul (.get "ft.pid_m" .zero) (.param .bm)) (.axis .index)),
  .integer "ft.rn" .n (.add (.mul (.get "ft.pid_n" .zero) (.param .bn)) (.axis .index)),
  .integer "ft.rk" .k (.axis .index),
  .pointer "ft.A" .mk (.add (.base .a) (.add (.mul (.get "ft.rm" .row) (.param .strideAM)) (.mul (.get "ft.rk" .inner) (.param .strideAK)))),
  .pointer "ft.B" .kn (.add (.base .b) (.add (.mul (.get "ft.rk" .inner) (.param .strideBK)) (.mul (.get "ft.rn" .col) (.param .strideBN)))),
  .zero "ft.acc" .mn,
  .forLoop "ft.k" (.lit 0) (.cdiv (.param .k) (.param .bk)) [
  .load "ft.a" .mk (.get "ft.A"),
  .load "ft.b" .kn (.get "ft.B"),
  .mac "ft.acc" "ft.a" "ft.b",
  .pointer "ft.A" .mk (.add (.get "ft.A") (.mul (.param .bk) (.param .strideAK))),
  .pointer "ft.B" .kn (.add (.get "ft.B") (.mul (.param .bk) (.param .strideBK)))
],
  .half "ft.acc" "ft.acc",
  .integer "ft.rm" .m (.add (.mul (.get "ft.pid_m" .zero) (.param .bm)) (.axis .index)),
  .integer "ft.rn" .n (.add (.mul (.get "ft.pid_n" .zero) (.param .bn)) (.axis .index)),
  .pointer "ft.C" .mn (.add (.base .c) (.add (.mul (.get "ft.rm" .row) (.param .strideCM)) (.mul (.get "ft.rn" .col) (.param .strideCN)))),
  .store .mn (.get "ft.C") "ft.acc"
]

end VeriTile.Bench.TritonBenchG.StreamK
