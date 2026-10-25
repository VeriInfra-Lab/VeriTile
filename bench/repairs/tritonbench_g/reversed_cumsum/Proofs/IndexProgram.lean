import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ReversedCumsum.IndexProgram
open VeriTile.Triton

def indexOp : Op .int [] := .sub .int .nil (.constInt 0) (.ref .int [] "__rev_i_t")
def startTOp (c : Config) : Op .int [] :=
  .mul .int .nil (.ref .int [] "i_t") (.constInt c.BT)
def startSOp : Op .int [] := .mul .int .nil (.ref .int [] "i_s") (.constInt (Int.ofNat BS))
def activeTOp (h : Call) (c : Config) : Op .bool [c.BT] :=
  .lt .int .scalarR (.castNatToInt (.ref .nat [c.BT] "lane_t"))
    (.sub .int .nil (.castNatToInt (.constNat h.T)) (.ref .int [] "start_t"))
def activeSOp (h : Call) : Op .bool [BS] :=
  .lt .int .scalarR (.castNatToInt (.ref .nat [BS] "lane_s"))
    (.sub .int .nil (.castNatToInt (.constNat h.S)) (.ref .int [] "start_s"))
def timeOp (c : Config) : Op .int [c.BT] :=
  .add .int .scalarL (.ref .int [] "start_t")
    (.castNatToInt (.where (.ref .bool [c.BT] "active_t") (.ref .nat [c.BT] "lane_t") (.broadcast (.constNat 0) [c.BT])))
def featureOp : Op .int [BS] :=
  .add .int .scalarL (.ref .int [] "start_s")
    (.castNatToInt (.where (.ref .bool [BS] "active_s") (.ref .nat [BS] "lane_s") (.broadcast (.constNat 0) [BS])))
def maskOp (c : Config) : Op .bool [c.BT,BS] :=
  .boolAnd (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .bool [c.BT] "active_t"))
    (.expandDim ⟨0,by simp⟩ (.ref .bool [BS] "active_s"))
def pointerOp (h : Call) (c : Config) (r : RegionName) : Op .ptr [c.BT,BS] :=
  .ptrAdd .scalarL (.ptrBase r) (.castIntToNat
    (.add .int (.consR (.consL .nil))
      (.add .int .scalarL
        (.mul .int .nil (.ref .int [] "i_bh") (.castNatToInt (.constNat (h.T*h.S))))
        (.mul .int .scalarR (.expandDim ⟨1,by simp⟩ (.ref .int [c.BT] "times")) (.castNatToInt (.constNat h.S))))
      (.mul .int .scalarR (.expandDim ⟨0,by simp⟩ (.ref .int [BS] "features"))
        (.castNatToInt (.constNat 1)))))
def stopOp (h : Call) (c : Config) : Op .nat [] :=
  .constNat (Geometry.chunks h c)
def reverseStart (h : Call) (c : Config) : Op .int [] :=
  .sub .int .nil (.constInt 0) (.sub .int .nil (.castNatToInt (stopOp h c)) (.constInt 1))
def nonnegativeOp : Op .bool [] :=
  .ge .int .nil (.ref .int [] "i_t") (.constInt 0)
end VeriTile.Bench.Repairs.ReversedCumsum.IndexProgram
