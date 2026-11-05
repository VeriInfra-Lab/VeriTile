import VeriTile.Triton

namespace VeriTile.Bench.Repairs.KvCacheCopy.Source
open VeriTile.Triton

structure Parameters where
  (strideKt strideKh strideKd strideVt strideVh strideVd
    strideKcb strideKch strideKcsplit strideKcs
    strideVcb strideVch strideVcs strideVcd strideBts strideBtb
    blockSize dims width : Nat)

end VeriTile.Bench.Repairs.KvCacheCopy.Source
