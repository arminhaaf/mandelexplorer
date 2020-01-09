#include	<jni.h>
#include	<stdlib.h>
#include <stdbool.h>
#ifndef	_Included_Mandel
#define	_Included_Mandel
#ifdef	__cplusplus
extern	"C"	{
#endif


JNIEXPORT	void	JNICALL	Java_nimra_mandelexplorer_MandelNative_mandel
		(JNIEnv	*,	jobject,
		const jint algo,
		unsigned jintArray iters,
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
        const jdouble sqrEscapeRadius);

#ifdef	__cplusplus
}
#endif
#endif