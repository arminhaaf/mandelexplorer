CC=clang

CFLAGS =-Wall -Wextra -O3 -fopenmp -I${JAVA_HOME}/include/ -I${JAVA_HOME}/include/linux -fPIC

libmandel_jni.so :  mandeljni.c mandelImpl.o mandelAVXDD.o mandelDD.o
	$(CC) $(CFLAGS) $(LDFLAGS) -lstdc++ -shared  -o $@ $^ $(LDLIBS)

mandelImpl.o : mandelImpl.c
	$(CC) -c $(CFLAGS) -Ofast -mavx2 -mfma -o $@ $^

mandelAVXDD.o : mandelAVXDD.c
	$(CC) -c $(CFLAGS) -mavx2 -ffp-contract=off -fno-unsafe-math-optimizations -o $@ $^

mandelDD.o : mandelDD.c
	$(CC) -c $(CFLAGS) -ffp-contract=off -fno-unsafe-math-optimizations  -o $@ $^

