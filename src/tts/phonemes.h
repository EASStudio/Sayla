#pragma once

#include <stdbool.h>

// A phoneme inventory with approximate formant characteristics, used by
// formant_synth.c to synthesize speech from scratch (no TTS library).
//
// The vowel formant values are typical/ballpark figures of the kind
// found in standard acoustic-phonetics vowel charts (e.g. the general
// shape of the classic Peterson & Barney measurements) -- not
// measurements calibrated to any specific voice. Expect this to sound
// clearly robotic and to only loosely resemble any one person's actual
// vowel space; that's an inherent property of formant synthesis at this
// level of simplicity, not a bug to chase.
//
// Consonants are grouped into a handful of broad acoustic categories
// (nasal, fricative, stop, liquid/glide) with one representative
// filter/noise treatment per category, rather than a fully accurate
// individual acoustic model of each -- a complete model of every
// English consonant's place and manner of articulation is a much
// bigger undertaking than fits here.

typedef enum
{
    PH_SILENCE,

    // Vowels
    PH_IY, // beet
    PH_IH, // bit
    PH_EH, // bet
    PH_AE, // bat
    PH_AA, // father
    PH_AO, // bought
    PH_UH, // book
    PH_UW, // boot
    PH_AH, // but
    PH_AX, // schwa, "a" in "about"

    // Nasals
    PH_M, PH_N, PH_NG,

    // Fricatives, unvoiced
    PH_S, PH_F, PH_SH, PH_TH, PH_H,
    // Fricatives, voiced
    PH_Z, PH_V, PH_ZH, PH_DH,

    // Stops (crude approximation: a brief burst, not a real
    // closure+release model)
    PH_P, PH_T, PH_K, PH_B, PH_D, PH_G,

    // Liquids / glides -- vowel-like, formant values approximate their
    // closest associated vowel quality
    PH_L, PH_R, PH_W, PH_Y,

    PHONEME_COUNT
} Phoneme;

typedef struct
{
    float f1, f2, f3;    // formant center frequencies, Hz
    float bw1, bw2, bw3; // formant bandwidths, Hz
    bool voiced;         // true = pulse-train (glottal) source, false = noise source
    float amplitude;     // relative loudness, 0..1
    int durationMs;      // typical duration when spoken at a neutral, unhurried pace
} PhonemeInfo;

extern const PhonemeInfo phonemeTable[PHONEME_COUNT];