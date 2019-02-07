import org.ejml.simple.SimpleMatrix
import java.util.*
import kotlin.concurrent.thread
import kotlin.math.exp

fun assert(a: Boolean) {
    if(!a) throw IllegalStateException()
}

fun doubleArr(length: Int, elem: Double) = DoubleArray(length, { elem })

fun Double.mult(x: SimpleMatrix): SimpleMatrix {
    return x.elementApply { e -> this * e }
}

class NNData(val layers: List<Int>) {
    val Os: Array<SimpleMatrix>

    val extendedWs: Array<SimpleMatrix>

    var Ds: Array<SimpleMatrix>
    lateinit var e: SimpleMatrix

    init {
        val rand = Random()

        Os = Array(layers.size) {
            emptySimpleMatrix()
        }

        extendedWs = Array(layers.size-1) { i ->
            SimpleMatrix.random_DDRM(layers[i]+1, layers[i+1], 0.0, 1.0 - Double.MIN_VALUE, rand)
        }

        Ds = Array(layers.size-1) { i ->
            emptySimpleMatrix()
        }
    }
}

const val c = 1
fun s(z: Double) = 1/(1 + exp(-c*z))

const val gamma: Double = 1.0

class NN(val net: NNData) {
    val deltaWs: Array<SimpleMatrix>

    init {
        deltaWs = Array(net.extendedWs.size) { i ->
            net.extendedWs[i].createLike()
        }
    }

    fun process(o: SimpleMatrix, t: SimpleMatrix): SimpleMatrix {
        var oTemp = o

        net.Os[0] = o

        for (i in 0 until net.extendedWs.size) {
            oTemp = oTemp.extend().rowWiseMult(net.extendedWs[i]).columnWiseSum().transpose().elementApply(::s)

            net.Ds[i] = SimpleMatrix(net.layers[i+1], 1, true, DoubleArray(net.layers[i+1]) { i ->
                ({ oi: Double -> oi*(1 - oi) })(oTemp[i, 0])
            })

            net.Os[i+1] = oTemp
        }

        net.e = oTemp.minus(t)

        return oTemp
    }

    fun backpropagate() {
        var delta = net.Ds[net.Ds.size-1].mult(net.e)

        for (i in (net.extendedWs.size-1) downTo 0) {
            val deltaW = (-gamma).mult(delta).mult(net.Os[i].extend().transpose()).transpose()

            if(i > 0) {
                delta = net.Ds[i-1].rowWiseMult(net.extendedWs[i].unextend().matrixVectorMult(delta))
            }

            deltaWs[i] = deltaWs[i].plus(deltaW)
        }
    }

    fun correct(): Pair<Double, Double> {
        val error = getError()

        for (i in 0 until net.extendedWs.size) {
            net.extendedWs[i] = net.extendedWs[i].plus(deltaWs[i])
        }

        for(deltaW in deltaWs) {
            for(i in 0 until deltaW.numRows()) {
                for (j in 0 until deltaW.numCols()) {
                    deltaW[i, j] = 0.0
                }
            }
        }

        return Pair(error, getError())
    }

    fun getError() = net.e.elementApply { (it * it)/2.0 }.elementSum()
}

fun getResult(net: NN, x: DoubleArray, y: DoubleArray): Double {
    val o = SimpleMatrix(x.size, 1, true, x)
    val t = SimpleMatrix(y.size, 1, true, y)

    return net.process(o, t)[0]
}

fun trainFor(net: NN, list: List<Pair<Pair<Double, Double>, Double>>) {
    for((input, result) in list) {
        val (x, y) = input
        val r = getResult(net, doubleArrayOf(x, y), doubleArrayOf(result))
        println("($x, $y) = ${r} (should be ${result})")
    }

    Runtime.getRuntime().addShutdownHook( thread(false) {
        println("------------------------")

        for((input, result) in list) {
            val (x, y) = input
            val r = getResult(net, doubleArrayOf(x, y), doubleArrayOf(result))
            println("($x, $y) = ${r}")
        }
    })

    for (i in 0 .. Long.MAX_VALUE) {
        for((input, result) in list) {
            val (x, y) = input

            getResult(net, doubleArrayOf(x, y), doubleArrayOf(result))
            net.backpropagate()
        }

        net.correct()

        var errorSum = 0.0

        for((input, result) in list) {
            val (x, y) = input

            getResult(net, doubleArrayOf(x, y), doubleArrayOf(result))

            errorSum += net.getError()
        }

        if(errorSum < 0.0001) break
        //println("${i}: $errorSum")
    }
}

fun main() {
    val layers = listOf(2, 3, 1)

    val net = NN(NNData(layers))

    trainFor(net, listOf(
        (0.0 to 0.0) to 0.0,
        (1.0 to 0.0) to 1.0,
        (0.0 to 1.0) to 1.0,
        (1.0 to 1.0) to 0.0
    ))
}