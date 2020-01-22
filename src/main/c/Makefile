.DEFAULT_GOAL := install

BASELIBNAME=mandel_jni
LIBNAME=

ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
endif

#CC=/usr/bin/x86_64-w64-mingw32-gcc
CC=clang

CFLAGS =-Wall -Wextra -O3 -fopenmp -I${JAVA_HOME}/include/  -fPIC

ifeq ($(detected_OS),Windows)
    CFLAGS += -I${JAVA_HOME}/include/win32
    LIBNAME=$(BASELIBNAME).dll
endif
ifeq ($(detected_OS),Darwin)        # Mac OS X
    CFLAGS += -I${JAVA_HOME}/include/darwin
endif
ifeq ($(detected_OS),Linux)
   CFLAGS   +=  -I${JAVA_HOME}/include/linux
    LIBNAME=lib$(BASELIBNAME).so
endif


install : $(LIBNAME)
	cp $(LIBNAME) ../resources/natives/linux_64

$(LIBNAME) :  mandeljni.c mandelImpl.o mandelAVXDD.o mandelDD.o mandelFloat128.o mandelFloat80.o
	$(CC) $(CFLAGS) $(LDFLAGS) -I. -lstdc++ -shared -D_JNI_IMPLEMENTATION_ -o $@ $^ $(LDLIBS)

mandelImpl.o : mandelImpl.c
	$(CC) -c $(CFLAGS) -Ofast -mavx2 -mfma -o $@ $^

mandelAVXDD.o : mandelAVXDD.c
	$(CC) -c $(CFLAGS) -mavx2 -ffp-contract=off -fno-unsafe-math-optimizations -o $@ $^

mandelDD.o : mandelDD.c
	$(CC) -c $(CFLAGS) -ffp-contract=off -fno-unsafe-math-optimizations  -o $@ $^

mandelFloat128.o : mandelFloat128.c
	$(CC) -c $(CFLAGS) -Ofast -o $@ $^

mandelFloat80.o : mandelFloat80.c
	$(CC) -c $(CFLAGS) -Ofast -o $@ $^

.PHONY: clean
clean:
	rm *.o

