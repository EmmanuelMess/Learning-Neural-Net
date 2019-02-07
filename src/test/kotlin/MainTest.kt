import org.ejml.EjmlUnitTests
import org.ejml.simple.SimpleMatrix
import org.hamcrest.MatcherAssert.assertThat
import org.hamcrest.core.Is
import org.hamcrest.core.IsEqual.equalTo
import org.junit.Test


class MainTest {
    @Test
    fun testExtend() {
        val o = SimpleMatrix(3, 1, true, doubleArrayOf(1.0, 2.0, 3.0))
        val t = SimpleMatrix(4, 1, true, doubleArrayOf(1.0, 2.0, 3.0, 1.0))

        EjmlUnitTests.assertEquals(o.extend().getMatrix(), t.getMatrix())
    }

    @Test
    fun testUnextend() {
        val o = SimpleMatrix(4, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0,
            1.0
        ))
        val t = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0
        ))

        EjmlUnitTests.assertEquals(o.unextend().getMatrix(), t.getMatrix())
    }

    @Test
    fun testRowWiseMult() {
        val x = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0))
        val y = SimpleMatrix(3, 2, true, doubleArrayOf(
            1.0, 2.0,
            3.0, 1.0,
            1.0, 1.0))

        val t = SimpleMatrix(3, 2, true, doubleArrayOf(
            1.0, 2.0,
            6.0, 2.0,
            3.0, 3.0
        ))

        EjmlUnitTests.assertEquals(x.rowWiseMult(y).getMatrix(), t.getMatrix())
    }

    @Test
    fun testColWiseSum() {
        val o = SimpleMatrix(2, 3, true, doubleArrayOf(1.0, 2.0, 3.0, 1.0, 1.0, 1.0))

        val t = SimpleMatrix(1, 3, true, doubleArrayOf(2.0, 3.0, 4.0))

        EjmlUnitTests.assertEquals(o.columnWiseSum().getMatrix(), t.getMatrix())
    }

    @Test
    fun testElementApply() {
        val o = SimpleMatrix(2, 3, true, doubleArrayOf(1.0, 2.0, 3.0, 1.0, 1.0, 1.0))

        val t = SimpleMatrix(2, 3, true, doubleArrayOf(2.0, 3.0, 4.0, 2.0, 2.0, 2.0))

        EjmlUnitTests.assertEquals(o.elementApply({ it+1 }).getMatrix(), t.getMatrix())
    }

    @Test
    fun testActivation() {
        val o = SimpleMatrix(2, 3, true, doubleArrayOf(
            1.0, 2.0, 3.0,
            1.0, 1.0, 1.0
        ))

        val t = SimpleMatrix(2, 3, true, doubleArrayOf(
            s(1.0),s( 2.0), s(3.0),
            s(1.0), s(1.0), s(1.0)
        ))

        EjmlUnitTests.assertEquals(o.elementApply(::s).getMatrix(), t.getMatrix())
    }

    @Test
    fun testMult() {
        val x = SimpleMatrix(2, 1, true, doubleArrayOf(
            1.0,
            2.0
        ))

        val y = SimpleMatrix(1, 3, true, doubleArrayOf(
            1.0, 2.0, 3.0
        ))

        val t = SimpleMatrix(2, 3, true, doubleArrayOf(
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0
        ))

        EjmlUnitTests.assertEquals(x.mult(y).getMatrix(), t.getMatrix())
    }

    @Test
    fun testConstantMult() {
        val x = 5.0

        val y = SimpleMatrix(1, 3, true, doubleArrayOf(
            1.0, 2.0, 3.0
        ))

        val t = SimpleMatrix(1, 3, true, doubleArrayOf(
            5.0, 10.0, 15.0
        ))

        EjmlUnitTests.assertEquals(x.mult(y).getMatrix(), t.getMatrix())
    }

    @Test
    fun testMatrixVectorMult() {
        val x = SimpleMatrix(2, 3, true, doubleArrayOf(
            1.0, 2.0, 3.0,
            2.0, 6.0, 8.0
        ))

        val y = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0
        ))

        val t = SimpleMatrix(2, 1, true, doubleArrayOf(
            14.0,
            2.0 + 12.0 + 24.0
        ))

        EjmlUnitTests.assertEquals(x.matrixVectorMult(y).getMatrix(), t.getMatrix())
    }

    @Test
    fun genericTest0() {
        val o = SimpleMatrix.diag(1.0, 2.0, 3.0)

        val d = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0
        ))

        val t = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0*1.0,
            2.0*2.0,
            3.0*3.0
        ))

        EjmlUnitTests.assertEquals(
            o.mult(d).getMatrix(),
            t.getMatrix()
        )
    }

    @Test
    fun genericTest1() {
        val gamma = 10.0

        val o = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0
        ))

        val d = SimpleMatrix(4, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0,
            4.0
        ))

        val t = SimpleMatrix(3, 4, true, doubleArrayOf(
            -gamma*1.0*1.0, -gamma*1.0*2.0, -gamma*1.0*3.0, -gamma*1.0*4.0,
            -gamma*2.0*1.0, -gamma*2.0*2.0, -gamma*2.0*3.0, -gamma*2.0*4.0,
            -gamma*3.0*1.0, -gamma*3.0*2.0, -gamma*3.0*3.0, -gamma*3.0*4.0
        ))

        EjmlUnitTests.assertEquals(
            (-gamma).mult(d).mult(o.transpose()).transpose().getMatrix(),
            t.getMatrix()
        )
    }

    @Test
    fun genericTest2() {
        val o = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0
        ))

        val w = SimpleMatrix(3, 3, true, doubleArrayOf(
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0,
            1.0, 2.0, 3.0
        ))

        val d = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0,
            2.0,
            3.0
        ))

        val t = SimpleMatrix(3, 1, true, doubleArrayOf(
            1.0*(1.0*1.0 + 2.0*2.0 + 3.0*3.0),
            2.0*(1.0*1.0 + 2.0*2.0 + 3.0*3.0),
            3.0*(1.0*1.0 + 2.0*2.0 + 3.0*3.0)
        ))

        EjmlUnitTests.assertEquals(o.rowWiseMult(w.matrixVectorMult(d)).getMatrix(), t.getMatrix())
    }
}