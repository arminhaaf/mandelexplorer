CFLAGS = -std=c99 -Wall -Wextra -Ofast -fopenmp -I/opt/jdk1.8/include/ -I/opt/jdk1.8/include/linux  -fPIC -shared 

libmandel_jni.so :  mandeljni.c mandelImpl.o  
	$(CC) $(CFLAGS) $(LDFLAGS) -fPIC -shared  -o $@ $^ $(LDLIBS) 

mandelImpl.o : mandelImpl.c
	$(CC) -c $(CFLAGS) -mavx -o $@ $^
