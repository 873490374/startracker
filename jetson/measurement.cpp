#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <time.h>

int main() {
    int fd = open("/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_voltage0_input", O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        perror("open()");
        exit(1);
    }
    int cnt = 0;
    double sum = 0;
    double start = 0;
    while (true) {
        if (cnt == 0) {
            struct timespec tv;
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv);
            start = tv.tv_sec + tv.tv_nsec * 1e-9;
        }
        char buf[31];
        lseek(fd, 0, 0);
        int n = read(fd, buf, 32);
        if (n > 0) {
            buf[n] = 0;
            char *o = NULL;
            sum += strtod(buf, &o);
            cnt += 1;
        }
        if (cnt >= 1000) {
            struct timespec tv;
            clock_gettime(CLOCK_MONOTONIC_RAW, &tv);
            double end = tv.tv_sec + tv.tv_nsec * 1e-9;
            fprintf(stderr, "Read %d values in %.3f milliseconds\n", cnt, (end - start) * 1000);
            fprintf(stderr, "average value was %.1f\n", sum / cnt);
            cnt = 0;
            sum = 0;
        }
    }
    return 0;
}

