//  hammingseg.h
//      
//	Copyright (C) 2011-2012  Signal processing laboratory 2, EPFL,
//	Kirell Benzi (kirell.benzi@epfl.ch),
//	Raphael Ortiz (raphael.ortiz@a3.epfl.ch),
//	Alexandre Alahi (alexandre.alahi@epfl.ch),
//	and Pierre Vandergheynst (pierre.vandergheynst@epfl.ch)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

#ifndef HAMMINGSEG_H_INCLUDED
#define HAMMINGSEG_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <stdint.h>

namespace cv {

#ifdef __GNUC__
static const char __attribute__((aligned(16))) MASK[16] = {0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf};
static const uint8_t __attribute__((aligned(16))) LUT_POP[16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
static const __m128i SHIFT = _mm_set_epi32(0,0,0,4);
#endif

#ifdef _MSC_VER
__declspec(align(16)) static const char MASK[16] = {0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf, 0xf};
__declspec(align(16)) static const uint8_t LUT_POP[16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
static const __m128i SHIFT = _mm_set_epi32(0,0,0,4);
#endif

/** Hamming distance using SSE instruction with intermediat threshold
     * @param Tr intermediate threshold on the first 128 bits
     * allows ~25% speed gain while not affecting the final result
     * too restrictive value will reject true positive
     * too high value will increase the computation time (no early rejection, but an extra condition tested anyway)
     * for optimal efficiency make sure the 512 description pairs have been selected accordingly to the detector
     * values out of range [0,128] have no effect
     * @param nbBlocks number of blocks of 128 bits
*/
template<int Tr = -1, int nbBlocks = 4>
struct CV_EXPORTS HammingSeg
{
    // 512 hamming distance segmented in 128+384 bits
    // SSSE3
    // adapted from http://wm.ite.pl/articles/sse-popcount.html
    // and BRISK: Binary Robust Invariant Scalable Keypoints : http://www.asl.ethz.ch/people/lestefan/personal/BRISK
    // http://en.wikipedia.org/wiki/Hamming_weight
    static CV_INLINE uint32_t XORedPopcnt_128_384( const __m128i* string1, const __m128i* string2 )
    {
        register __m128i xmm0;
        register __m128i xmm1;
        register __m128i xmm2; // vector of popcount for lower nibbles
        register __m128i xmm3; // vector of popcount for higher nibbles
        register __m128i xmm4; // local accumulator (for inner loop)
        register __m128i xmm5; // global accumulator
        register __m128i xmm6; // MASK
        register __m128i xmm7; // LUT_POP
        register __m128i xmm8;

        xmm7 = _mm_load_si128((__m128i *)LUT_POP); // Returns the value loaded in a variable representing a register
        xmm6 = _mm_load_si128((__m128i *)MASK);
        xmm5 = _mm_setzero_si128(); // Sets the 128-bit value to zero
        xmm4 = xmm5;
        // 4 times unrolled loop (512 bits) {
        xmm0 = _mm_xor_si128( (__m128i)*string1++, (__m128i)*string2++); // Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in b.
        xmm1 = xmm0;
        xmm1 = _mm_srl_epi16(xmm1, SHIFT); // Shifts the 8 signed or unsigned 16-bit integers in a right by count bits while shifting in zeros.
        xmm0 = _mm_and_si128(xmm0, xmm6); // Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b.
        xmm1 = _mm_and_si128(xmm1, xmm6);
        xmm2 = xmm7;
        xmm3 = xmm7;
        xmm2 = _mm_shuffle_epi8(xmm2, xmm0); // lower nibbles.Emits the Supplemental Streaming SIMD Extensions 3 (SSSE3) instruction pshufb.
        // This instruction shuffles 16-byte parameters from a 128-bit parameter.
        xmm3 = _mm_shuffle_epi8(xmm3, xmm1); // higher nibbles
        xmm4 = _mm_add_epi8(xmm4, xmm2); // Adds the 16 signed or unsigned 8-bit integers in a to the 16 signed or unsigned 8-bit integers in b.
        xmm4 = _mm_add_epi8(xmm4, xmm3);

        /////////////////////
        if( Tr < 128 && Tr > 0 ) {
            xmm8 = _mm_sad_epu8(xmm4, xmm5);
            //~ xmm0 = _mm_alignr_epi8( xmm0,xmm8,8 );
            xmm0 = (__m128i) _mm_movehl_ps( (__m128) xmm0, (__m128) xmm8);
            xmm0 = _mm_add_epi32(xmm0, xmm8);
            if( _mm_cvtsi128_si32(xmm0) > Tr ) {
                return 999999999;
            }
        }
        /////////////////////

        xmm0 = _mm_xor_si128( (__m128i)*string1++, (__m128i)*string2++); // Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in b.
        xmm1 = xmm0;
        xmm1 = _mm_srl_epi16(xmm1, SHIFT); // Shifts the 8 signed or unsigned 16-bit integers in a right by count bits while shifting in zeros.
        xmm0 = _mm_and_si128(xmm0, xmm6); // Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b.
        xmm1 = _mm_and_si128(xmm1, xmm6);
        xmm2 = xmm7;
        xmm3 = xmm7;
        xmm2 = _mm_shuffle_epi8(xmm2, xmm0);
        xmm3 = _mm_shuffle_epi8(xmm3, xmm1); // higher nibbles
        xmm4 = _mm_add_epi8(xmm4, xmm2); // Adds the 16 signed or unsigned 8-bit integers in a to the 16 signed or unsigned 8-bit integers in b.
        xmm4 = _mm_add_epi8(xmm4, xmm3);

        xmm0 = _mm_xor_si128( (__m128i)*string1++, (__m128i)*string2++); //Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in b.
        xmm1 = xmm0;
        xmm1 = _mm_srl_epi16(xmm1, SHIFT); // Shifts the 8 signed or unsigned 16-bit integers in a right by count bits while shifting in zeros.
        xmm0 = _mm_and_si128(xmm0, xmm6); // Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b.
        xmm1 = _mm_and_si128(xmm1, xmm6);
        xmm2 = xmm7;
        xmm3 = xmm7;
        xmm2 = _mm_shuffle_epi8(xmm2, xmm0);
        xmm3 = _mm_shuffle_epi8(xmm3, xmm1); // higher nibbles
        xmm4 = _mm_add_epi8(xmm4, xmm2); // Adds the 16 signed or unsigned 8-bit integers in a to the 16 signed or unsigned 8-bit integers in b.
        xmm4 = _mm_add_epi8(xmm4, xmm3);

        xmm0 = _mm_xor_si128( (__m128i)*string1++, (__m128i)*string2++); // Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in b.
        xmm1 = xmm0;
        xmm1 = _mm_srl_epi16(xmm1, SHIFT); // Shifts the 8 signed or unsigned 16-bit integers in a right by count bits while shifting in zeros.
        xmm0 = _mm_and_si128(xmm0, xmm6); // Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b.
        xmm1 = _mm_and_si128(xmm1, xmm6);
        xmm2 = xmm7;
        xmm3 = xmm7;
        xmm2 = _mm_shuffle_epi8(xmm2, xmm0); // lower nibbles
        xmm3 = _mm_shuffle_epi8(xmm3, xmm1); //higher nibbles
        xmm4 = _mm_add_epi8(xmm4, xmm2); //Adds the 16 signed or unsigned 8-bit integers in a to the 16 signed or unsigned 8-bit integers in b.
        xmm4 = _mm_add_epi8(xmm4, xmm3);

        if( nbBlocks > 4 ) {
            for( int i = nbBlocks-4; i--; ) {
                xmm0 = _mm_xor_si128( (__m128i)*string1++, (__m128i)*string2++); // Computes the bitwise XOR of the 128-bit value in a and the 128-bit value in b.
                xmm1 = xmm0;
                xmm1 = _mm_srl_epi16(xmm1, SHIFT); //Shifts the 8 signed or unsigned 16-bit integers in a right by count bits while shifting in zeros.
                xmm0 = _mm_and_si128(xmm0, xmm6); // Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b.
                xmm1 = _mm_and_si128(xmm1, xmm6);
                xmm2 = xmm7;
                xmm3 = xmm7;
                xmm2 = _mm_shuffle_epi8(xmm2, xmm0); // lower nibbles
                xmm3 = _mm_shuffle_epi8(xmm3, xmm1); // higher nibbles
                xmm4 = _mm_add_epi8(xmm4, xmm2); // Adds the 16 signed or unsigned 8-bit integers in a to the 16 signed or unsigned 8-bit integers in b.
                xmm4 = _mm_add_epi8(xmm4, xmm3);
            }
        }
        ////}

        xmm4 = _mm_sad_epu8(xmm4, xmm5); // Computes the absoLUT_POPe difference of the 16 unsigned 8-bit integers from a and the 16 unsigned 8-bit integers from b.
        // Sums the upper 8 differences and lower 8 differences and packs the resulting 2 unsigned 16-bit integers into the upper and lower 64-bit elements.

        xmm0 = (__m128i) _mm_movehl_ps( (__m128) xmm0, (__m128) xmm4);
        xmm0 = _mm_add_epi32(xmm0, xmm4);
        return _mm_cvtsi128_si32(xmm0); // Moves the least significant 32 bits of a to a 32-bit integer.
    }

    enum {normType =  NORM_HAMMING};
    typedef unsigned char ValueType;
    //! important that this is signed as weird behavior happens
    // in BruteForce if not
    typedef int ResultType;

    // 512bits block bit count with optional early stop after first 128 bits  block
    ResultType operator()( const unsigned char* a, const unsigned char* b, const int size ) const {
        return XORedPopcnt_128_384((const __m128i*)a, (const __m128i*)b);
    }
};
} // END NAMESPACE CV

#endif // HAMMINGSEG_H_INCLUDED
