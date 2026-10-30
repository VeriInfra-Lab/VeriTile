import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.IndexProgram
open VeriTile.Triton

def startOp (nm : RegName) (width : Nat) : Op .int [] :=
  .mul .int .nil (.ref .int [] nm) (.constInt width)
def timeStartOp (c : Config) : Op .int [] :=
  .mul .int .nil (.ref .int [] "i_t") (.castNatToInt (.constNat c.BT))
def activeOp (length width : Nat) (lanes start : RegName) : Op .bool [width] :=
  .lt .int .scalarR (.ref .int [width] lanes)
    (.sub .int .nil (.castNatToInt (.constNat length)) (.ref .int [] start))
def coordinateOp (width : Nat) (start active lanes : RegName) : Op .int [width] :=
  .add .int .scalarL (.ref .int [] start)
    (.where (.ref .bool [width] active) (.ref .int [width] lanes)
      (.broadcast (.constInt 0) [width]))
def maskOp (c : Config) : Op .bool [c.BT,BS] :=
  .boolAnd (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .bool [c.BT] "active_t"))
    (.expandDim ⟨0,by simp⟩ (.ref .bool [BS] "active_s"))
def pointerOp (c : Config) (r : RegionName) (strides : Strides) : Op .ptr [c.BT,BS] :=
  .ptrAdd .scalarL (.ptrBase r) (.castIntToNat
    (.add .int (.consR (.consL .nil))
      (.add .int .scalarL
        (.add .int .nil
          (.mul .int .nil (.ref .int [] "i_b") (.castNatToInt (.constNat strides.batch)))
          (.mul .int .nil (.ref .int [] "i_h") (.castNatToInt (.constNat strides.head))))
        (.mul .int .scalarR (.expandDim ⟨1,by simp⟩ (.ref .int [c.BT] "times"))
          (.castNatToInt (.constNat strides.time))))
      (.mul .int .scalarR (.expandDim ⟨0,by simp⟩ (.ref .int [BS] "columns"))
        (.castNatToInt (.constNat strides.channel)))))
def countOp (h : Call) (c : Config) : Op .nat [] :=
  .constNat (Geometry.chunks h c)
def triangularOp (c : Config) : Op .real [c.BT,c.BT] :=
  .where (.ge .nat (.consR (.consL .nil))
    (.expandDim ⟨1,by simp⟩ (.ref .nat [c.BT] "o_i"))
    (.expandDim ⟨0,by simp⟩ (.ref .nat [c.BT] "o_i")))
    (.broadcast (.const 1.0) [c.BT,c.BT]) (.broadcast (.const 0.0) [c.BT,c.BT])
def outputOp (c : Config) : Op .real [c.BT,BS] :=
  .add .real (.consL (.consSame .nil))
    (.expandDim ⟨0,by simp⟩ (.ref .real [BS] "b_z"))
    (.dot (batch:=[]) (.ref .real [c.BT,c.BT] "m_s") (.ref .real [c.BT,BS] "b_s")
      {allowTF32:=some Bool.false})
def carryOp (c : Config) : Op .real [BS] :=
  .add .real (.consSame .nil) (.ref .real [BS] "b_z")
    (.reduceSum ⟨0,by simp⟩ Bool.false (.ref .real [c.BT,BS] "b_s"))
end VeriTile.Bench.Repairs.ChunkCumsumVector.IndexProgram
