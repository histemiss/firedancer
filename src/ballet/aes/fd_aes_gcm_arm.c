#include "fd_aes_gcm.h"

/* fd_aes_gcm_arm.c: AES-GCM backend using the ARMv8 Crypto Extensions
   (FEAT_AES) for the AES block cipher and FEAT_PMULL for GHASH.

   The ARM AESE/AESMC instructions use the natural (FIPS-197) byte order
   and the semantics
       AESE(x, k) = ShiftRows(SubBytes(x XOR k))
       AESMC(x)   = MixColumns(x)
   so the AES-128/192/256 encryption is:
       s = AESE(p, rk[0])
       for r in 1..Nr-1: s = AESMC(AESE(s, rk[r]))
       s = s XOR rk[Nr]

   The round keys are produced here in natural byte order (byte 0 = MSB
   of each FIPS-197 word).  Do not reuse fd_aes_set_encrypt_key, which
   stores u32 round keys in the host (little-endian) byte order.

   GHASH uses PMULL (carry-less multiply) in the natural bit order
   (bit i = coefficient x^i), with the reduction polynomial
   x^128 + x^7 + x^2 + x + 1 (fold: x^128 = x^7+x^2+x+1).  The GCM spec
   stores coefficients in descending order (b_0 = x^127), so each block is
   bit-reversed within each byte (RBIT) to reach natural order before the
   multiply, and bit-reversed back after.  This produces byte-identical
   results to the portable 4-bit table GHASH (fd_gcm_ghash_4bit). */

#if FD_HAS_ARM

#include <arm_neon.h>

/* AES S-box (FIPS-197). */
static uchar const fd_aes_arm_sbox[ 256 ] = {
  0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
  0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
  0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
  0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
  0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
  0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
  0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
  0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
  0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
  0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
  0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
  0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
  0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
  0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
  0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
  0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

/* Round constants Rcon[i] = x^(i-1) in GF(2^8), stored as the low byte. */
static uchar const fd_aes_arm_rcon[ 10 ] = {
  0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

/* fd_aes_arm_set_encrypt_key: standard FIPS-197 key schedule, producing
   (nr+1) 16-byte round keys in natural byte order. */

static void
fd_aes_arm_set_encrypt_key( uchar const *     key,
                            ulong             key_sz,   /* 16/24/32 */
                            uchar             rk[ 15 ][ 16 ],
                            uint *            nr ) {    /* 10/12/14 */

  uint nk = (uint)( key_sz >> 2 );        /* 4, 6, 8 words */
  uint nr_ = nk + 6;                      /* 10, 12, 14 */
  uint nb = 4;
  uchar w[ 60 ][ 4 ];                     /* up to 60 words (AES-256) */

  for( uint i=0; i<nk; i++ )
    for( uint j=0; j<4; j++ )
      w[ i ][ j ] = key[ 4*i + j ];

  for( uint i=nk; i<nb*(nr_+1); i++ ) {
    uchar temp[ 4 ];
    for( uint j=0; j<4; j++ ) temp[ j ] = w[ i-1 ][ j ];
    if( ( i % nk )==0 ) {
      /* RotWord */
      uchar t = temp[ 0 ];
      temp[ 0 ] = temp[ 1 ]; temp[ 1 ] = temp[ 2 ]; temp[ 2 ] = temp[ 3 ]; temp[ 3 ] = t;
      /* SubWord */
      for( uint j=0; j<4; j++ ) temp[ j ] = fd_aes_arm_sbox[ temp[ j ] ];
      /* Rcon */
      temp[ 0 ] ^= fd_aes_arm_rcon[ i/nk - 1 ];
    } else if( nk>6 && ( i % nk )==4 ) {
      /* AES-256: SubWord on every 4th word */
      for( uint j=0; j<4; j++ ) temp[ j ] = fd_aes_arm_sbox[ temp[ j ] ];
    }
    for( uint j=0; j<4; j++ ) w[ i ][ j ] = w[ i-nk ][ j ] ^ temp[ j ];
  }

  for( uint r=0; r<=nr_; r++ )
    for( uint j=0; j<4; j++ )
      for( uint c=0; c<4; c++ )
        rk[ r ][ 4*j + c ] = w[ 4*r + j ][ c ];

  *nr = nr_;
}

/* fd_aes_arm_encrypt_block: single-block AES encryption via AESE/AESMC. */

__attribute__((target("+crypto")))
static void
fd_aes_arm_encrypt_block( uchar const       in [ 16 ],
                          uchar             out[ 16 ],
                          uchar             rk[ 15 ][ 16 ],
                          uint              nr ) {
  uint8x16_t s = vld1q_u8( in );
  s = vaeseq_u8( s, vld1q_u8( rk[ 0 ] ) );
  for( uint r=1; r<nr; r++ ) {
    s = vaesmcq_u8( s );
    s = vaeseq_u8( s, vld1q_u8( rk[ r ] ) );
  }
  s = veorq_u8( s, vld1q_u8( rk[ nr ] ) );
  vst1q_u8( out, s );
}

/* --- GHASH via PMULL ---------------------------------------------------- */

/* fd_ulong_rbit: full 64-bit bit reversal (RBIT instruction). */
static inline ulong
fd_ulong_rbit( ulong x ) {
  ulong r;
  __asm__ volatile( "rbit %0, %1" : "=r"(r) : "r"(x) );
  return r;
}

/* fd_ulong_bitrev8: reverse the bits within each byte of a 64-bit word.
   RBIT gives full 64-bit reversal; a byte swap on top leaves only the
   per-byte bit reversal (RBIT and byte-swap commute). */
static inline ulong
fd_ulong_bitrev8( ulong x ) {
  return fd_ulong_bswap( fd_ulong_rbit( x ) );
}

/* fd_gcm_pmull_mul: GF(2^128) multiply in natural bit order (bit i = x^i),
   reduction polynomial x^128 + x^7 + x^2 + x + 1.  Operands are (lo,hi)
   64-bit halves; result returned in (*r_lo,*r_hi).  Computed as a 4-PMULL
   schoolbook carry-less product followed by the fold reduction
       Z = P_lo ^ P_hi ^ (P_hi << 1) ^ (P_hi << 2) ^ (P_hi << 7)
   iterated until the result fits in 128 bits (converges in 2 iterations). */
__attribute__((target("+crypto")))
static inline void
fd_gcm_pmull_mul( ulong  a_lo, ulong  a_hi,
                  ulong  b_lo, ulong  b_hi,
                  ulong * r_lo, ulong * r_hi ) {
  uint128 p00 = (uint128)vmull_p64( (poly64_t)a_lo, (poly64_t)b_lo );
  uint128 p01 = (uint128)vmull_p64( (poly64_t)a_lo, (poly64_t)b_hi );
  uint128 p10 = (uint128)vmull_p64( (poly64_t)a_hi, (poly64_t)b_lo );
  uint128 p11 = (uint128)vmull_p64( (poly64_t)a_hi, (poly64_t)b_hi );

  ulong w0 = (ulong)p00;
  ulong w1 = (ulong)( p00 >> 64 ) ^ (ulong)p01 ^ (ulong)p10;
  ulong w2 = (ulong)( p01 >> 64 ) ^ (ulong)( p10 >> 64 ) ^ (ulong)p11;
  ulong w3 = (ulong)( p11 >> 64 );

  ulong z0 = w0, z1 = w1, z2 = w2, z3 = w3;
  while( z2 | z3 ) {
    ulong h0 = z2, h1 = z3;   /* high 128 bits = [h1:h0] */
    z0 ^= h0 ^ ( h0 << 1 ) ^ ( h0 << 2 ) ^ ( h0 << 7 );
    z1 ^= h1 ^ ( ( h1 << 1 ) | ( h0 >> 63 ) )
             ^ ( ( h1 << 2 ) | ( h0 >> 62 ) )
             ^ ( ( h1 << 7 ) | ( h0 >> 57 ) );
    z2 = ( h1 >> 63 ) ^ ( h1 >> 62 ) ^ ( h1 >> 57 );
    z3 = 0;
  }
  *r_lo = z0;
  *r_hi = z1;
}

/* fd_gcm_gmult_pmull: Xi = Xi * H in GF(2^128).  Xi is a 16-byte block in
   GCM byte order (ulong[2], little-endian halves); H_nat is the hash key in
   natural order ([0]=lo, [1]=hi). */
static void
fd_gcm_gmult_pmull( ulong       Xi[ 2 ],
                    ulong const H_nat[ 2 ] ) {
  ulong a_lo = fd_ulong_bitrev8( Xi[ 0 ] );
  ulong a_hi = fd_ulong_bitrev8( Xi[ 1 ] );
  ulong r_lo, r_hi;
  fd_gcm_pmull_mul( a_lo, a_hi, H_nat[ 0 ], H_nat[ 1 ], &r_lo, &r_hi );
  Xi[ 0 ] = fd_ulong_bitrev8( r_lo );
  Xi[ 1 ] = fd_ulong_bitrev8( r_hi );
}

/* fd_gcm_ghash_pmull: GHASH over len bytes (multiple of 16): for each block,
   Xi = (Xi ^ block) * H.  Xi and H_nat as in fd_gcm_gmult_pmull. */
static void
fd_gcm_ghash_pmull( ulong             Xi[ 2 ],
                    ulong const       H_nat[ 2 ],
                    uchar const *     inp,
                    ulong             len ) {
  ulong a_lo = fd_ulong_bitrev8( Xi[ 0 ] );
  ulong a_hi = fd_ulong_bitrev8( Xi[ 1 ] );
  while( len > 0 ) {
    ulong b0 = fd_ulong_load_8( inp     );
    ulong b1 = fd_ulong_load_8( inp + 8 );
    a_lo ^= fd_ulong_bitrev8( b0 );
    a_hi ^= fd_ulong_bitrev8( b1 );
    fd_gcm_pmull_mul( a_lo, a_hi, H_nat[ 0 ], H_nat[ 1 ], &a_lo, &a_hi );
    inp += 16;
    len -= 16;
  }
  Xi[ 0 ] = fd_ulong_bitrev8( a_lo );
  Xi[ 1 ] = fd_ulong_bitrev8( a_hi );
}

/* ARM GCM state: same GCM fields as the portable reference, but with the
   ARM round keys instead of fd_aes_key_ref_t.  (State layout defined in
   fd_aes_gcm.h.) */

static void
fd_aes_gcm_arm_setiv( fd_aes_gcm_arm_t * gcm,
                      uchar const        iv[ 12 ] ) {
  uint ctr;
  gcm->len.u[ 0 ] = 0;
  gcm->len.u[ 1 ] = 0;
  gcm->ares = 0;
  gcm->mres = 0;

  memcpy( gcm->Yi.c, iv, 12 );
  gcm->Yi.c[12] = 0;
  gcm->Yi.c[13] = 0;
  gcm->Yi.c[14] = 0;
  gcm->Yi.c[15] = 1;
  ctr = 1;

  gcm->Xi.u[0] = 0;
  gcm->Xi.u[1] = 0;

  fd_aes_arm_encrypt_block( gcm->Yi.c, gcm->EK0.c, gcm->rk, gcm->nr );
  ctr++;

  gcm->Yi.d[3] = fd_uint_bswap( ctr );
}

void
fd_aes_gcm_init_arm( fd_aes_gcm_arm_t * gcm,
                     uchar const *      key,
                     ulong              key_sz,
                     uchar const        iv[ 12 ] ) {
  memset( gcm, 0, sizeof( fd_aes_gcm_arm_t ) );

  fd_aes_arm_set_encrypt_key( key, key_sz, gcm->rk, &gcm->nr );

  fd_aes_arm_encrypt_block( (uchar const *)gcm->H.c, gcm->H.c, gcm->rk, gcm->nr );
  /* gcm->H.c was zeroed by memset; encrypt the all-zero block to get H in
     GCM (NIST) byte order. */

  gcm->H_nat[ 0 ] = fd_ulong_bitrev8( gcm->H.u[ 0 ] );
  gcm->H_nat[ 1 ] = fd_ulong_bitrev8( gcm->H.u[ 1 ] );

  fd_aes_gcm_arm_setiv( gcm, iv );
}

/* --- GCM mode: mirrors fd_aes_gcm_ref.c with fd_aes_arm_encrypt_block --- */

static int
fd_gcm128_arm_aad( fd_aes_gcm_arm_t * gcm,
                   uchar const *      aad,
                   ulong              aad_sz ) {
  ulong alen = gcm->len.u[ 0 ];
  if( FD_UNLIKELY( gcm->len.u[ 1 ] ) ) return -2;
  alen += aad_sz;
  if( alen > (1UL<<61) || ( sizeof(aad_sz)==8 && alen < aad_sz ) ) return -1;
  gcm->len.u[0] = alen;

  uint n = gcm->ares;
  if( n ) {
    while( n && aad_sz ) {
      gcm->Xi.c[n] ^= *(aad++);
      --aad_sz;
      n = (n + 1) % 16;
    }
    if( n == 0 ) fd_gcm_gmult_pmull( gcm->Xi.u, gcm->H_nat );
    else { gcm->ares = n; return 0; }
  }
  ulong i;
  if( ( i = ( aad_sz & (ulong)-16 ) ) ) {
    fd_gcm_ghash_pmull( gcm->Xi.u, gcm->H_nat, aad, i );
    aad += i;
    aad_sz -= i;
  }
  if( aad_sz ) {
    n = (uint)aad_sz;
    for( i=0; i<aad_sz; ++i ) gcm->Xi.c[i] ^= aad[i];
  }
  gcm->ares = n;
  return 0;
}

static int
fd_gcm128_arm_crypt( fd_aes_gcm_arm_t * gcm,
                     uchar const *      in,
                     uchar *            out,
                     ulong              len,
                     int                enc ) {
  uint n, ctr, mres;
  ulong i;
  ulong mlen = gcm->len.u[1];

  mlen += len;
  if( mlen > ((1UL<<36) - 32) || ( sizeof(len)==8 && mlen < len ) ) return -1;
  gcm->len.u[1] = mlen;

  mres = gcm->mres;

  if( gcm->ares ) {
    if( len == 0 ) { fd_gcm_gmult_pmull( gcm->Xi.u, gcm->H_nat ); gcm->ares = 0; return 0; }
    memcpy( gcm->Xn, gcm->Xi.c, sizeof(gcm->Xi) );
    gcm->Xi.u[0] = 0;
    gcm->Xi.u[1] = 0;
    mres = (uint)sizeof(gcm->Xi);
    gcm->ares = 0;
  }

  ctr = fd_uint_bswap( gcm->Yi.d[3] );

  n = mres % 16;
  for( i=0; i<len; ++i ) {
    uchar c;
    if( n == 0 ) {
      fd_aes_arm_encrypt_block( gcm->Yi.c, gcm->EKi.c, gcm->rk, gcm->nr );
      ++ctr;
      gcm->Yi.d[3] = fd_uint_bswap( ctr );
    }
    if( enc ) {
      c = in[i] ^ gcm->EKi.c[n];
      out[i] = gcm->Xn[mres++] = c;
    } else {
      c = in[i];
      gcm->Xn[mres++] = c;
      out[i] = c ^ gcm->EKi.c[n];
    }
    n = (n + 1) % 16;
    if( mres == sizeof(gcm->Xn) ) {
      fd_gcm_ghash_pmull( gcm->Xi.u, gcm->H_nat, gcm->Xn, sizeof(gcm->Xn) );
      mres = 0;
    }
  }

  gcm->mres = mres;
  return 0;
}

static void
fd_gcm128_arm_finish( fd_aes_gcm_arm_t * gcm ) {
  ulong alen = gcm->len.u[0] << 3;
  ulong clen = gcm->len.u[1] << 3;
  struct { ulong hi; ulong lo; } bitlen;
  uint mres = gcm->mres;

  if( mres ) {
    uint blocks = ( mres + 15u ) & 0xfffffff0u;
    memset( gcm->Xn + mres, 0, blocks - mres );
    mres = blocks;
    if( mres == sizeof(gcm->Xn) ) {
      fd_gcm_ghash_pmull( gcm->Xi.u, gcm->H_nat, gcm->Xn, mres );
      mres = 0;
    }
  } else if( gcm->ares ) {
    fd_gcm_gmult_pmull( gcm->Xi.u, gcm->H_nat );
  }

  alen = fd_ulong_bswap( alen );
  clen = fd_ulong_bswap( clen );
  bitlen.hi = alen;
  bitlen.lo = clen;
  memcpy( gcm->Xn + mres, &bitlen, sizeof(bitlen) );
  mres += (uint)sizeof(bitlen);
  fd_gcm_ghash_pmull( gcm->Xi.u, gcm->H_nat, gcm->Xn, mres );

  gcm->Xi.u[0] ^= gcm->EK0.u[0];
  gcm->Xi.u[1] ^= gcm->EK0.u[1];
}

void
fd_aes_gcm_encrypt_arm( fd_aes_gcm_arm_t * gcm,
                        uchar *            c,
                        uchar const *      p,
                        ulong              sz,
                        uchar const *      aad,
                        ulong              aad_sz,
                        uchar              tag[ 16 ] ) {
  fd_gcm128_arm_aad( gcm, aad, aad_sz );
  int res = fd_gcm128_arm_crypt( gcm, p, c, sz, 1 );
  FD_DCHECK_CRIT( res==0, "internal error" );
  fd_gcm128_arm_finish( gcm );
  fd_memcpy( tag, gcm->Xi.c, 16 );
}

int
fd_aes_gcm_decrypt_arm( fd_aes_gcm_arm_t * gcm,
                        uchar const *      c,
                        uchar *            p,
                        ulong              sz,
                        uchar const *      aad,
                        ulong              aad_sz,
                        uchar const        tag[ 16 ] ) {
  fd_gcm128_arm_aad( gcm, aad, aad_sz );
  int res = fd_gcm128_arm_crypt( gcm, c, p, sz, 0 );
  FD_DCHECK_CRIT( res==0, "internal error" );
  fd_gcm128_arm_finish( gcm );
  return 0 == memcmp( gcm->Xi.c, tag, 16 );
}

#else /* !FD_HAS_ARM */

/* Avoid an empty translation unit on non-ARM targets. */
int fd_aes_gcm_arm_dummy;

#endif /* FD_HAS_ARM */
