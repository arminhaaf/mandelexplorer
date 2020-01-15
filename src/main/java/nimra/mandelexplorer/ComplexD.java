package nimra.mandelexplorer;

import java.util.Objects;

/**
 * Created: 29.12.19   by: Armin Haaf
 *
 * adapted from https://introcs.cs.princeton.edu/java/32class/Complex.java
 *
 * @author Armin Haaf
 */
public class ComplexD {
    public double re;   // the real part
    public double im;   // the imaginary part

    public ComplexD(double real, double imag) {
        re = real;
        im = imag;
    }

    public ComplexD(final ComplexD other) {
        this.re = other.re;
        this.im = other.im;
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

    // return a new Complex object whose value is (this * b)
    public ComplexD mul(ComplexD b) {
        final double tmp = re * b.re - im * b.im;
        im = re * b.im + im * b.re;
        re = tmp;
        return this;
    }

    // return a new object whose value is (this * alpha)
    public ComplexD scale(double alpha) {
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

//
//    // return a new Complex object whose value is the complex sine of this
//    public ComplexD sin() {
//        return new ComplexD(Math.sin(re) * Math.cosh(im), Math.cos(re) * Math.sinh(im));
//    }
//
//    // return a new Complex object whose value is the complex cosine of this
//    public ComplexD cos() {
//        return new ComplexD(Math.cos(re) * Math.cosh(im), -Math.sin(re) * Math.sinh(im));
//    }
//
//    // return a new Complex object whose value is the complex tangent of this
//    public ComplexD tan() {
//        return sin().div(cos());
//    }


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
