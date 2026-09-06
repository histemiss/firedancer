/* Benchmark + stress test for ARM crypto backends vs portable reference.
   Links against libfd_ballet.a. */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "src/ballet/aes/fd_aes_gcm.h"
#include "src/ballet/sha256/fd_sha256.h"
#include "src/ballet/sha512/fd_sha512.h"

/* ref backend prototypes are not exported from fd_aes_gcm_ref.h */
void fd_aes_gcm_init_ref( fd_aes_gcm_ref_t * gcm, uchar const * key, ulong key_sz, uchar const iv[12] );
void fd_aes_gcm_encrypt_ref( fd_aes_gcm_ref_t * aes_gcm, uchar * c, uchar const * p, ulong sz, uchar const * aad, ulong aad_sz, uchar tag[16] );
int  fd_aes_gcm_decrypt_ref( fd_aes_gcm_ref_t * aes_gcm, uchar const * c, uchar * p, ulong sz, uchar const * aad, ulong aad_sz, uchar const tag[16] );

static inline ulong now_ns(void) {
  struct timespec ts; clock_gettime( CLOCK_MONOTONIC, &ts );
  return (ulong)ts.tv_sec * 1000000000UL + (ulong)ts.tv_nsec;
}

/* xorshift64* PRNG for reproducible random data */
static ulong rng_state = 0x9e3779b97f4a7c15UL;
static inline ulong rnd(void) {
  ulong x = rng_state;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  rng_state = x;
  return x * 0x2545F4914F6CDD1DUL;
}
static void fill_rand( uchar * p, ulong n ) {
  for( ulong i=0; i<n; i++ ) p[i] = (uchar)( rnd() >> 32 );
}

static volatile ulong sink = 0; /* prevent optimization */

int main( void ) {

  /* ---- 1. Correctness stress: AES-GCM arm vs ref ---- */
  printf("== AES-GCM correctness stress (arm vs ref) ==\n");
  int fails = 0;
  ulong N = 100000;
  for( ulong t=0; t<N; t++ ) {
    ulong key_sz = ( t % 3 == 0 ) ? 16UL : ( t % 3 == 1 ? 24UL : 32UL );
    uchar key[32]; fill_rand( key, key_sz );
    uchar iv[12];  fill_rand( iv, 12 );
    ulong sz   = rnd() % 512;
    ulong aad_sz = rnd() % 64;
    uchar p[512], c1[512], c2[512], aad[64];
    fill_rand( p, sz ); fill_rand( aad, aad_sz );
    uchar tag1[16], tag2[16];

    fd_aes_gcm_t arm[1] __attribute__((aligned(64)));
    fd_aes_gcm_ref_t ref[1] __attribute__((aligned(64)));
    fd_aes_gcm_init_arm( arm, key, key_sz, iv );
    fd_aes_gcm_init_ref( ref, key, key_sz, iv );
    fd_aes_gcm_encrypt_arm( arm, c1, p, sz, aad, aad_sz, tag1 );
    fd_aes_gcm_encrypt_ref( ref, c2, p, sz, aad, aad_sz, tag2 );

    if( memcmp( c1, c2, sz ) || memcmp( tag1, tag2, 16 ) ) {
      fails++;
      if( fails < 4 ) printf("  MISMATCH at iter %lu (sz=%lu key_sz=%lu)\n", t, sz, key_sz );
    }
    /* decrypt roundtrip on ARM */
    uchar p2[512];
    fd_aes_gcm_init_arm( arm, key, key_sz, iv );
    if( !fd_aes_gcm_decrypt_arm( arm, c1, p2, sz, aad, aad_sz, tag1 ) ||
        memcmp( p2, p, sz ) ) {
      fails++;
      if( fails < 4 ) printf("  DECRYPT FAIL at iter %lu\n", t );
    }
  }
  printf("  %lu iterations, %d failures -> %s\n\n", N, fails, fails==0 ? "PASS" : "FAIL" );

  /* ---- 2. AES-GCM throughput: arm vs ref ---- */
  printf("== AES-128-GCM throughput (128-byte msgs) ==\n");
  for( int mode = 0; mode < 2; mode++ ) {
    uchar key[16] = {0}, iv[12] = {0};
    fd_aes_gcm_t arm[1] __attribute__((aligned(64)));
    fd_aes_gcm_ref_t ref[1] __attribute__((aligned(64)));
    fd_aes_gcm_init_arm( arm, key, 16, iv );
    fd_aes_gcm_init_ref( ref, key, 16, iv );
    uchar p[128], c[128], tag[16];
    memset( p, 0xab, 128 );
    ulong iters = 2000000;
    ulong t0 = now_ns();
    if( mode == 0 ) for( ulong i=0; i<iters; i++ ) fd_aes_gcm_encrypt_arm( arm, c, p, 128, NULL, 0, tag );
    else            for( ulong i=0; i<iters; i++ ) fd_aes_gcm_encrypt_ref( ref, c, p, 128, NULL, 0, tag );
    ulong dt = now_ns() - t0;
    double mbs = (double)iters * 128.0 / 1024.0 / 1024.0 / ( (double)dt / 1e9 );
    printf("  %-4s: %7.1f MB/s  (%lu iters, %.3f s)\n", mode==0?"ARM":"ref", mbs, iters, (double)dt/1e9 );
    sink += c[0];
  }
  printf("\n");

  /* ---- 3. AES-256-GCM throughput ---- */
  printf("== AES-256-GCM throughput (128-byte msgs) ==\n");
  for( int mode = 0; mode < 2; mode++ ) {
    uchar key[32] = {0}, iv[12] = {0};
    fd_aes_gcm_t arm[1] __attribute__((aligned(64)));
    fd_aes_gcm_ref_t ref[1] __attribute__((aligned(64)));
    fd_aes_gcm_init_arm( arm, key, 32, iv );
    fd_aes_gcm_init_ref( ref, key, 32, iv );
    uchar p[128], c[128], tag[16];
    memset( p, 0xab, 128 );
    ulong iters = 2000000;
    ulong t0 = now_ns();
    if( mode == 0 ) for( ulong i=0; i<iters; i++ ) fd_aes_gcm_encrypt_arm( arm, c, p, 128, NULL, 0, tag );
    else            for( ulong i=0; i<iters; i++ ) fd_aes_gcm_encrypt_ref( ref, c, p, 128, NULL, 0, tag );
    ulong dt = now_ns() - t0;
    double mbs = (double)iters * 128.0 / 1024.0 / 1024.0 / ( (double)dt / 1e9 );
    printf("  %-4s: %7.1f MB/s\n", mode==0?"ARM":"ref", mbs );
    sink += c[0];
  }
  printf("\n");

  /* ---- 4. SHA256 / SHA512 throughput (active ARM path) ---- */
  printf("== SHA256 (ARM crypto ext) throughput: 32-byte msgs ==\n");
  {
    uchar msg[32], out[32]; memset( msg, 0x42, 32 );
    ulong iters = 5000000;
    ulong t0 = now_ns();
    for( ulong i=0; i<iters; i++ ) fd_sha256_hash( msg, 32, out );
    ulong dt = now_ns() - t0;
    double mhs = (double)iters / ( (double)dt / 1e9 ) / 1e6;
    printf("  %7.1f Mhash/s\n", mhs );
    sink += out[0];
  }
  printf("== SHA512 (ARM crypto ext) throughput: 64-byte msgs ==\n");
  {
    uchar msg[64], out[64]; memset( msg, 0x42, 64 );
    ulong iters = 5000000;
    ulong t0 = now_ns();
    for( ulong i=0; i<iters; i++ ) fd_sha512_hash( msg, 64, out );
    ulong dt = now_ns() - t0;
    double mhs = (double)iters / ( (double)dt / 1e9 ) / 1e6;
    printf("  %7.1f Mhash/s\n", mhs );
    sink += out[0];
  }

  printf("\nsink=%lu (ignore)\n", (ulong)sink );
  return fails == 0 ? 0 : 1;
}
