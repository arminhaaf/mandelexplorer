#include	<jni.h>
#include	<stdlib.h>
#include    <stdbool.h>
#include    "mandel.h"
#include    "mandelAVXDD.h"
#ifndef	_Included_MadelNative
#define	_Included_MadelNative

#ifdef	__cplusplus
extern	"C"	{
#endif

JNIEXPORT	void	JNICALL	Java_nimra_mandelexplorer_DDMandelNative_mandelDD
		(JNIEnv	*env,	jobject obj,
		const jint algo,
		jintArray iters,
		jdoubleArray lastZrs,
   		jdoubleArray lastZis,
		jdoubleArray distancesR,
		jdoubleArray distancesI,
		const int mode,
		const jint width,
        const jint height,
        const jdouble xStartHi,
        const jdouble xStartLo,
        const jdouble yStartHi,
        const jdouble yStartLo,
        const jdouble juliaCrHi,
        const jdouble juliaCrLo,
        const jdouble juliaCiHi,
        const jdouble juliaCiLo,
        const jdouble xIncHi,
        const jdouble xIncLo,
        const jdouble yIncHi,
        const jdouble yIncLo,
        const jint maxIterations,
        const jdouble sqrEscapeRadius) {

        int32_t* tIters = (*env)->GetIntArrayElements(env, iters, 0);
        double* tLastZrs = (*env)->GetDoubleArrayElements(env, lastZrs, 0);
        double* tLastZis = (*env)->GetDoubleArrayElements(env, lastZis, 0);
        double* tDistancesR = (*env)->GetDoubleArrayElements(env, distancesR, 0);
        double* tDistancesI = (*env)->GetDoubleArrayElements(env, distancesI, 0);

//        printf("%.16e %.16e - %.16e %.16e\n", xIncHi, xIncLo, xIncHi, yIncLo);
//        fflush(stdout);

        switch ( algo ) {
            case 1:
                    mandel_avxdd(tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                        width, height,
                        xStartHi, xStartLo, yStartHi, yStartLo,
                        juliaCrHi, juliaCrLo, juliaCiHi, juliaCiLo,
                        xIncHi, xIncLo, yIncHi, yIncLo,
                        maxIterations,sqrEscapeRadius);
                break;
            case 2:
                    mandel_dd(tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                        width, height,
                        xStartHi, xStartLo, yStartHi, yStartLo,
                        juliaCrHi, juliaCrLo, juliaCiHi, juliaCiLo,
                        xIncHi, xIncLo, yIncHi, yIncLo,
                        maxIterations,sqrEscapeRadius);
                break;
            case 3:
                mandel_float128(tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                        width, height,
                        xStartHi, xStartLo, yStartHi, yStartLo,
                        juliaCrHi, juliaCrLo, juliaCiHi, juliaCiLo,
                        xIncHi, xIncLo, yIncHi, yIncLo,
                        maxIterations,sqrEscapeRadius);
                break;
            case 4:
                mandel_float80(tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                        width, height,
                        xStartHi, xStartLo, yStartHi, yStartLo,
                        juliaCrHi, juliaCrLo, juliaCiHi, juliaCiLo,
                        xIncHi, xIncLo, yIncHi, yIncLo,
                        maxIterations,sqrEscapeRadius);
                break;
            default:
                printf("unknown algorithm %d", algo);
                fflush(stdout);
                break;


        }


    (*env)->ReleaseIntArrayElements(env, iters, tIters, 0);
    (*env)->ReleaseDoubleArrayElements(env, lastZrs, tLastZrs, 0);
    (*env)->ReleaseDoubleArrayElements(env, lastZis, tLastZis, 0);
    (*env)->ReleaseDoubleArrayElements(env, distancesR, tDistancesR, 0);
    (*env)->ReleaseDoubleArrayElements(env, distancesI, tDistancesI, 0);

}



JNIEXPORT	void	JNICALL	Java_nimra_mandelexplorer_DoubleMandelNative_mandel
		(JNIEnv	*env,	jobject obj,
		const jint algo,
		jintArray iters,
		jdoubleArray lastZrs,
   		jdoubleArray lastZis,
		jdoubleArray distancesR,
		jdoubleArray distancesI,
		const jint mode,
		const jint width,
        const jint height,
        const jdouble xStart,
        const jdouble yStart,
        const jdouble juliaCr,
        const jdouble juliaCi,
        const jdouble xInc,
        const jdouble yInc,
        const jint maxIterations,
        const jdouble sqrEscapeRadius) {

        jint* tIters = (*env)->GetIntArrayElements(env, iters, 0);
        double* tLastZrs = (*env)->GetDoubleArrayElements(env, lastZrs, 0);
        double* tLastZis = (*env)->GetDoubleArrayElements(env, lastZis, 0);
        double* tDistancesR = (*env)->GetDoubleArrayElements(env, distancesR, 0);
        double* tDistancesI = (*env)->GetDoubleArrayElements(env, distancesI, 0);
        switch ( algo ) {
            case 1:
                mandel_avxd(tIters, tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                    width, height, xStart, yStart, juliaCr, juliaCi, xInc, yInc, maxIterations,sqrEscapeRadius);
                break;
            case 2:
                mandel_avxs(tIters, tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                    width, height, xStart, yStart, juliaCr, juliaCi, xInc, yInc, maxIterations,sqrEscapeRadius);
                break;
            case 3:
                mandel_double(tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, mode,
                    width, height, xStart, yStart, juliaCr, juliaCi, xInc, yInc, maxIterations,sqrEscapeRadius);
                break;
        }


    (*env)->ReleaseIntArrayElements(env, iters, tIters, 0);
    (*env)->ReleaseDoubleArrayElements(env, lastZrs, tLastZrs, 0);
    (*env)->ReleaseDoubleArrayElements(env, lastZis, tLastZis, 0);
    (*env)->ReleaseDoubleArrayElements(env, distancesR, tDistancesR, 0);
    (*env)->ReleaseDoubleArrayElements(env, distancesI, tDistancesI, 0);

}




#ifdef	__cplusplus
}
#endif
#endif