.DEFAULT_GOAL := install

BASELIBNAME=mandel_jni
LIBNAME=
TARGETDIR=../resources/natives

CC=clang

ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
    detected_OS := Windows
else
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
endif

CFLAGS =-Wall -Wextra -O3 -fopenmp -I${JAVA_HOME}/include/  -fPIC

ifeq ($(detected_OS),Windows)
    CFLAGS += -I${JAVA_HOME}/include/win32 -fopenmp
    LDFLAGS += -static-libgcc -Wl,--add-stdcall-alias -Wl,-Bstatic -lgomp -lpthread
    LIBNAME=$(BASELIBNAME).dll
    TARGETDIR=../resources/natives/windows_64
    # currently only mingw is supported
    CC=cc
endif
ifeq ($(detected_OS),Darwin)        # Mac OS X currently not supported
    CFLAGS += -I${JAVA_HOME}/include/darwin
endif
ifeq ($(detected_OS),Linux)
   CFLAGS   +=  -I${JAVA_HOME}/include/linux
   LIBNAME=lib$(BASELIBNAME).so
   TARGETDIR=../resources/natives/linux_64
endif


install : $(LIBNAME)
	cp $(LIBNAME) $(TARGETDIR)

$(LIBNAME) :  mandeljni.o mandelDouble.o mandelAVXS.o mandelAVXD.o mandelAVXDD.o mandelDD.o mandelFloat128.o mandelFloat80.o  mandelSSEDD.o
	$(CC) $(CFLAGS) $(LDFLAGS) -I. -shared -D_JNI_IMPLEMENTATION_ -o $@ $^ $(LDLIBS)

mandeljni.o : mandeljni.c
	$(CC) -c $(CFLAGS) -Ofast -o $@ $^

mandelAVXD.o : mandelAVXD.c
	$(CC) -c $(CFLAGS) -Ofast -mavx2 -mfma -o $@ $^

mandelAVXS.o : mandelAVXS.c
	$(CC) -c $(CFLAGS) -Ofast -mavx2 -mfma -o $@ $^

mandelAVXDD.o : mandelAVXDD.c
	$(CC) -c $(CFLAGS) -mavx2 -mfma -fno-unsafe-math-optimizations -o $@ $^

mandelSSEDD.o : mandelSSEDD.c
	$(CC) -c $(CFLAGS) -msse2 -mfma -fno-unsafe-math-optimizations -o $@ $^

mandelSSED.o : mandelSSED.c
	$(CC) -c $(CFLAGS) -Ofast -msse2 -mfma -o $@ $^

mandelDD.o : mandelDD.c
	$(CC) -c $(CFLAGS) -Ofast -fno-unsafe-math-optimizations  -o $@ $^

mandelDouble.o : mandelDouble.c
	$(CC) -c $(CFLAGS) -Ofast -o $@ $^

mandelFloat128.o : mandelFloat128.c
	$(CC) -c $(CFLAGS) -Ofast -o $@ $^

mandelFloat80.o : mandelFloat80.c
	$(CC) -c $(CFLAGS) -Ofast -o $@ $^

.PHONY: clean
clean:
	rm *.o
	rm $(LIBNAME)

