CFLAGS =-Wall -Wextra -Ofast -fopenmp -I/opt/jdk1.8/include/ -I/opt/jdk1.8/include/linux -fPIC

libmandel_jni.so :  mandeljni.c mandelImpl.o MandelVectorAVX.o
	$(CC) $(CFLAGS) $(LDFLAGS) -lstdc++ -shared  -o $@ $^ $(LDLIBS)

mandelImpl.o : mandelImpl.c
	$(CC) -c $(CFLAGS) -mavx -o $@ $^

MandelVectorAVX.o : MandelVector.cpp
	$(CC) -std=c++17 -c $(CFLAGS) -mavx -o $@ $^
