
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;


/**
 * @author astepanov
 */
public class VarProximityTests {

    private final static double AVG_EPS = 1.e-5;

    private final static double P_0 = 0.95;

    private final double sigma_0;

    private final double p_eps;
    private final double sigma_eps;

    // H0 : sigma_{1, 2} = sigma_0 * (1 + eps_{1, 2}),
    // eps ~ TSP(p_eps), E(eps) = 0, Var(eps) = sigma_eps

    public static enum TEST{FISHER, LINK};

    private final TEST testType;

    private final ExecutorService executor;

    public VarProximityTests(double sigma_0,
                             double p_eps,
                             double sigma_eps,
                             TEST   testType,
                             int    nThreads) {

        this.sigma_0   = sigma_0;
        this.p_eps     = p_eps;
        this.sigma_eps = sigma_eps;
        this.testType  = testType;

        executor = Executors.newFixedThreadPool(nThreads);
    }

    private double[] getSample(int n, double eps) {

        ThreadLocalRandom rnd = ThreadLocalRandom.current();

        double x[] = new double[n];
        for (int i = 0; i < n; ++i) { 
            x[i] = sigma_0 * (1. + eps) * rnd.nextGaussian();
        }
        return x;
    }

    // p > 0 => TSP
    // p < 0 => normal
    private double[] getEpsSample(int n) {

        ThreadLocalRandom rnd = ThreadLocalRandom.current();

        double e[] = new double[n];

        if (p_eps < 0.) {
            for (int i = 0; i < n; ++i) { e[i] = sigma_eps * rnd.nextGaussian(); }
            return e;
        }

        double c = 0.5;

        // make the variance equal to sigma_eps
        double var = 0.5 / ((p_eps + 1.) * (p_eps + 2.));
        double kv = sigma_eps / Math.sqrt(var);

        for (int i = 0; i < n; ++i) {
            double u = rnd.nextDouble();
            double v;
            if (u < c) {
                v = c * Math.pow(u / c, 1. / p_eps);
            } else {
                v = 1. - (1. - c) * Math.pow((1. - u) / (1. - c), 1. / p_eps);
            }
            e[i] = kv * (v - c);
        }

        return e;
    }

    private double getS2(double x[]) {

        int nx = x.length;
        if (nx < 3) { throw new IllegalArgumentException("invalid nx: " + nx); }

        double m = 0., s = 0.;
        for (double v: x) { m += v; }
        m /= nx;

        for (double v: x) { s += (v - m) * (v - m); }

        return s / (nx - 1.);
    }

    private double getRange(double x[]) {

        int nx = x.length;
        if (nx < 3) { throw new IllegalArgumentException("invalid nx: " + nx); }

        double minx = x[0], maxx = x[0];
        for (int i = 1; i < nx; ++i) {
            double t = x[i];
            if (t < minx) { minx = t; }
            else if (t > maxx) { maxx = t; }
        }

        return maxx - minx;
    }

    private double getR(double x1[], double x2[]) {

        if (testType == TEST.LINK) {
            return getRange(x1) / getRange(x2);
        } else {
            return getS2(x1) / getS2(x2);
        }
    }


    private double[] getCriticalVals(int n1, int n2, int nSim) {

        double r[] = new double[nSim];

        double eps1[] = getEpsSample(nSim);
        double eps2[] = getEpsSample(nSim);

        for (int i = 0; i < nSim; ++i) {

            double x1[] = getSample(n1, eps1[i]);
            double x2[] = getSample(n2, eps2[i]);

            r[i] = getR(x1, x2);
        }

        Arrays.sort(r);

        double d = 0.5 * (1. - P_0);
        int i1 = (int) (d * nSim), i2 = (int) ((1 - d) * nSim);

        double res[] = new double[2];
        res[0] = r[i1 - 1];
        res[1] = r[i2 - 1];
        return res;
    }

    public double[] getCriticalVals(int n1, int n2, int nSim, int nAvg) {

        AtomicLong s1 = new AtomicLong(0), s2 = new AtomicLong(0);

        CountDownLatch latch = new CountDownLatch(nAvg);

        double k = (int) (1. / AVG_EPS);

        for (int i = 0; i < nAvg; ++i) {

            executor.execute(() -> {

                double c[] = getCriticalVals(n1, n2, nSim);

                s1.addAndGet((int) (k * c[0]));
                s2.addAndGet((int) (k * c[1]));

                latch.countDown();
            });
        }

        try { latch.await(); }
        catch (InterruptedException ie) { throw new RuntimeException(ie); }

        double c1 = AVG_EPS * s1.doubleValue() / nAvg;
        double c2 = AVG_EPS * s2.doubleValue() / nAvg;

        return new double[]{c1, c2};
    }

    private double getP(int n1, int n2, double C1, double C2, int nSim) {

        double P = nSim;

        double eps1[] = getEpsSample(nSim);
        double eps2[] = getEpsSample(nSim);

        for (int i = 0; i < nSim; ++i) {

            double x1[] = getSample(n1, eps1[i]);
            double x2[] = getSample(n2, eps2[i]);

            double r = getR(x1, x2);
            if ((r < C1) || (r > C2)) { --P; }
        }

        return P / nSim;
    }

    public double getP(int n1, int n2, double C1, double C2, int nSim, int nAvg) {

        AtomicLong s = new AtomicLong(0);

        CountDownLatch latch = new CountDownLatch(nAvg);

        int k = (int) (1. / AVG_EPS);

        for (int i = 0; i < nAvg; ++i) {

            double p = getP(n1, n2, C1, C2, nSim);
            s.addAndGet((int) (k * p));

            latch.countDown();
        }

        try { latch.await(); }
        catch (InterruptedException ie) { throw new RuntimeException(ie); }

        return AVG_EPS * s.doubleValue() / nAvg;
    }

    public void shutdownPool() { executor.shutdown(); }


    public static void main(String args[]) {

        double sigma_0 = 1.;
        double sigma_eps = 0.20;
        System.out.println("sigma_eps = " + sigma_eps);

        int nThreads = Runtime.getRuntime().availableProcessors() - 2;
        int nSim = 1_000_000, nAvg = 300;

        int N[] = {10, 15, 20};
        double P_eps[] = {0.7, 1., 2., 3., 10., -1.};

        for (int n: N) {

            System.out.println("");
            System.out.println("n = " + n);

            for (double p_eps: P_eps) {

                VarProximityTests calculator = new VarProximityTests(
                    sigma_0,
                    p_eps,
                    sigma_eps,
                    TEST.LINK,
                    nThreads);

                double c[] = calculator.getCriticalVals(n, n, nSim, nAvg);
                double c1 = c[0], c2 = c[1];

                // check (must be ~ P_0)
                double P = calculator.getP(n, n, c1, c2, nSim, 10);
                if (Math.abs(P - P_0) > 1.e-3) {
                    System.err.println("WARN: P = " + P);
                }

                System.out.printf("%.3f\t%.3f\n", c1, c2);

                calculator.shutdownPool();
            }
        }
    }
}
