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

JNIEXPORT	void	JNICALL	Java_nimra_mandelexplorer_MandelDDNative_mandelDD
		(JNIEnv	*env,	jobject obj,
		const jint algo,
		jintArray iters,
		jdoubleArray lastZrs,
   		jdoubleArray lastZis,
		jdoubleArray distancesR,
		jdoubleArray distancesI,
		const bool calcDistance,
		const jint width,
        const jint height,
        const jdouble xStartHi,
        const jdouble xStartLo,
        const jdouble yStartHi,
        const jdouble yStartLo,
        const jdouble xIncHi,
        const jdouble xIncLo,
        const jdouble yIncHi,
        const jdouble yIncLo,
        const jint maxIterations,
        const jdouble sqrEscapeRadius) {

        int* tIters = (*env)->GetIntArrayElements(env, iters, 0);
        double* tLastZrs = (*env)->GetDoubleArrayElements(env, lastZrs, 0);
        double* tLastZis = (*env)->GetDoubleArrayElements(env, lastZis, 0);
        double* tDistancesR = (*env)->GetDoubleArrayElements(env, distancesR, 0);
        double* tDistancesI = (*env)->GetDoubleArrayElements(env, distancesI, 0);

//        printf("%.16e %.16e - %.16e %.16e\n", xIncHi, xIncLo, xIncHi, yIncLo);
//        fflush(stdout);

        switch ( algo ) {
            case 1:
                    mandel_avxdd((unsigned int*)tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, calcDistance,
                        width, height,
                        xStartHi, xStartLo, yStartHi, yStartLo,
                        xIncHi, xIncLo, yIncHi, yIncLo,
                        maxIterations,sqrEscapeRadius);
                break;
            case 2:
                    mandel_dd((unsigned int*)tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, calcDistance,
                        width, height,
                        xStartHi, xStartLo, yStartHi, yStartLo,
                        xIncHi, xIncLo, yIncHi, yIncLo,
                        maxIterations,sqrEscapeRadius);
                break;
        }


    (*env)->ReleaseIntArrayElements(env, iters, tIters, 0);
    (*env)->ReleaseDoubleArrayElements(env, lastZrs, tLastZrs, 0);
    (*env)->ReleaseDoubleArrayElements(env, lastZis, tLastZis, 0);
    (*env)->ReleaseDoubleArrayElements(env, distancesR, tDistancesR, 0);
    (*env)->ReleaseDoubleArrayElements(env, distancesI, tDistancesI, 0);

}



JNIEXPORT	void	JNICALL	Java_nimra_mandelexplorer_MandelNative_mandel
		(JNIEnv	*env,	jobject obj,
		const jint algo,
		jintArray iters,
		jdoubleArray lastZrs,
   		jdoubleArray lastZis,
		jdoubleArray distancesR,
		jdoubleArray distancesI,
		const bool calcDistance,
		const jint width,
        const jint height,
        const jdouble xStart,
        const jdouble yStart,
        const jdouble xInc,
        const jdouble yInc,
        const jint maxIterations,
        const jdouble sqrEscapeRadius) {

        int* tIters = (*env)->GetIntArrayElements(env, iters, 0);
        double* tLastZrs = (*env)->GetDoubleArrayElements(env, lastZrs, 0);
        double* tLastZis = (*env)->GetDoubleArrayElements(env, lastZis, 0);
        double* tDistancesR = (*env)->GetDoubleArrayElements(env, distancesR, 0);
        double* tDistancesI = (*env)->GetDoubleArrayElements(env, distancesI, 0);
        switch ( algo ) {
            case 1:
                mandel_avxd((unsigned int*)tIters, tLastZrs, tLastZis, tDistancesR, tDistancesI, calcDistance,
                    width, height, xStart, yStart, xInc, yInc, maxIterations,sqrEscapeRadius);
                break;
            case 2:
                mandel_avxs((unsigned int*)tIters, tLastZrs, tLastZis, tDistancesR, tDistancesI, calcDistance,
                    width, height, xStart, yStart, xInc, yInc, maxIterations,sqrEscapeRadius);
                break;
            case 3:
                mandel_double((unsigned int*)tIters,tLastZrs, tLastZis, tDistancesR, tDistancesI, calcDistance,
                    width, height, xStart, yStart, xInc, yInc, maxIterations,sqrEscapeRadius);
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