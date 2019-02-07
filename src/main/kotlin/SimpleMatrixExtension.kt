import org.ejml.data.DMatrixRMaj
import org.ejml.simple.SimpleMatrix

fun emptySimpleMatrix()
        = SimpleMatrix(Array(1) {
    DoubleArray(1) {
        0.0
    }
})

fun SimpleMatrix.extend(): SimpleMatrix {
    val result = copy()
    result.getMatrix<DMatrixRMaj>().reshape(numRows()+1, numCols(), true)
    result.setRow(result.numRows()-1, 0, *doubleArr(result.numCols(), 1.0 ))
    return result
}

fun SimpleMatrix.unextend(): SimpleMatrix {
    val result = copy()
    result.getMatrix<DMatrixRMaj>().reshape(numRows()-1, numCols(), true)
    return result
}

fun SimpleMatrix.matrixVectorMult(o: SimpleMatrix): SimpleMatrix {
    assert(o.numCols() == 1)
    assert(numCols() == o.numRows())

    val result = SimpleMatrix(numRows(), 1)

    for(i in 0 until numRows()) {
        var r = 0.0

        for(j in 0 until numCols()) {
            r += get(i, j) * o[j, 0]
        }

        result[i, 0] = r
    }

    return result
}

fun SimpleMatrix.rowWiseMult(o: SimpleMatrix): SimpleMatrix {
    assert(numCols() == 1)
    assert(numRows() == o.numRows())

    val result = o.createLike()

    for(i in 0 until numRows()) {
        for(j in 0 until o.numCols()) {
            result[i, j] = get(i, 0) * o[i, j]
        }
    }

    return result
}

fun SimpleMatrix.columnWiseSum(): SimpleMatrix {
    val result = SimpleMatrix(1, numCols())

    for(i in 0 until numCols()) {
        result[0, i] = cols(i , i+1).elementSum()
    }

    return result
}

fun SimpleMatrix.elementApply(f: (Double) -> Double): SimpleMatrix {
    val result = createLike()

    for(i in 0 until numRows()) {
        for(j in 0 until numCols()) {
            result.set(i, j, f(get(i, j)))
        }
    }

    return result
}