package nimra.mandelexplorer;

import java.util.Objects;

/**
 * Created: 29.12.19   by: Armin Haaf
 * <p>
 * adapted from https://introcs.cs.princeton.edu/java/32class/Complex.java
 *
 * @author Armin Haaf
 */
public class ComplexD {
    public double re;   // the real part
    public double im;   // the imaginary part

    public ComplexD() {
    }

    public ComplexD(double real, double imag) {
        re = real;
        im = imag;
    }

    public ComplexD(final ComplexD other) {
        this.re = other.re;
        this.im = other.im;
    }

    public ComplexD set(final ComplexD other) {
        this.re = other.re;
        this.im = other.im;
        return this;
    }

    public ComplexD set(final double pRe, final double pIm) {
        this.re = pRe;
        this.im = pIm;
        return this;
    }

    // return a string representation of the invoking Complex object
    public String toString() {
        if (im == 0) {
            return re + "";
        }
        if (re == 0) {
            return im + "i";
        }
        if (im < 0) {
            return re + " - " + (-im) + "i";
        }
        return re + " + " + im + "i";
    }

    // return abs/modulus/magnitude
    public double abs() {
        return Math.hypot(re, im);
    }

    public double magn() {
        return re * re + im * im;
    }

    // return angle/phase/argument, normalized to be between -pi and pi
    public double phase() {
        return Math.atan2(im, re);
    }

    public ComplexD add(ComplexD b) {
        re += b.re;
        im += b.im;
        return this;
    }

    // return a new Complex object whose value is (this + b)
    public ComplexD sub(ComplexD b) {
        re -= b.re;
        im -= b.im;
        return this;
    }

    // return a new Complex object whose value is (this + b)
    public ComplexD sub(double b) {
        re -= b;
        return this;
    }

    // return a new Complex object whose value is (this + b)
    public ComplexD add(double b) {
        re += b;
        return this;
    }

    // return a new Complex object whose value is (this * b)
    public ComplexD mul(ComplexD b) {
        final double tmp = re * b.re - im * b.im;
        im = re * b.im + im * b.re;
        re = tmp;
        return this;
    }

    public ComplexD sqr() {
        return mul(this);
    }

    // return a new object whose value is (this * alpha)
    public ComplexD mul(double alpha) {
        re *= alpha;
        im *= alpha;
        return this;
    }

    // return a new Complex object whose value is the conjugate of this
    public ComplexD conjugate() {
        im = -im;
        return this;
    }

    // return a new Complex object whose value is the reciprocal of this
    public ComplexD reciprocal() {
        final double scale = re * re + im * im;
        re /= scale;
        im /= -scale;
        return this;
    }

    // return the real or imaginary part
    public double re() {
        return re;
    }

    public double im() {
        return im;
    }

    // return a / b
    public ComplexD div(ComplexD b) {
        return mul(new ComplexD(b).reciprocal());
    }

    //    // return a new Complex object whose value is the complex exponential of this
    public ComplexD exp() {
        double tmp = Math.exp(re) * Math.cos(im);
        im = Math.exp(re) * Math.sin(im);
        re = tmp;
        return this;
    }

    public ComplexD sin() {
        double tmp = Math.sin(re) * Math.cosh(im);
        im = Math.cos(re) * Math.sinh(im);
        re = tmp;
        return this;
    }

    public ComplexD cos() {
        double tmp = Math.cos(re) * Math.cosh(im);
        im = -Math.sin(re) * Math.sinh(im);
        re = tmp;
        return this;
    }

    public ComplexD tan() {
        return sin().div(cos());
    }

    public ComplexD pow(double pDouble) {
        return log().mul(pDouble).exp();
    }

    public ComplexD pow(ComplexD b) {
        return log().mul(b).exp();
    }

    public ComplexD log() {
        re = Math.log(abs());
        im = Math.atan2(im, re);
        return this;
    }


    public ComplexD sqrt() {
        if (re == 0.0 && im == 0.0) {
            return this;
        }

        double t = Math.sqrt((Math.abs(re) + abs()) / 2.0);
        if (re >= 0.0) {
            re = t;
            im = im/(2.0*t);
        } else {
            re = Math.abs(im) / (2.0*t);
            im = Math.copySign(1.0d, im) * t;
        }
        return this;
    }
    public static ComplexD add(ComplexD a, ComplexD b) {
        return new ComplexD(a).add(b);
    }

    public static ComplexD sub(ComplexD a, ComplexD b) {
        return new ComplexD(a).sub(b);
    }

    public static ComplexD mul(ComplexD a, ComplexD b) {
        return new ComplexD(a).mul(b);
    }

    public static ComplexD div(ComplexD a, ComplexD b) {
        return new ComplexD(a).div(b);
    }

    public boolean equals(Object x) {
        if (x == null) {
            return false;
        }
        if (this.getClass() != x.getClass()) {
            return false;
        }
        ComplexD that = (ComplexD)x;
        return (this.re == that.re) && (this.im == that.im);
    }

    public int hashCode() {
        return Objects.hash(re, im);
    }
}
