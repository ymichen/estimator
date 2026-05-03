CC = gcc
CFLAGS = -fPIC -O3 -I .
TARGET = libpolar.so
SRCS = interface_implem.c

all: $(TARGET)

$(TARGET): $(SRCS) interface.h gaussian_channel.h polar_code.h ring.h tools.h
	$(CC) $(CFLAGS) -shared -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
