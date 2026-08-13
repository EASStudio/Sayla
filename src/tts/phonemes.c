// Module includes
#include "phonemes.h"

// Designated initializers throughout (not positional) so a reordering of
// the Phoneme enum in phonemes.h can never silently misalign an entry
// with the wrong phoneme.
const PhonemeInfo phonemeTable[PHONEME_COUNT] = {

    [PH_SILENCE] = { 0, 0, 0,  0, 0, 0,  false, 0.0f, 80 },

    // ---- Vowels ----
    // F1/F2/F3 follow the general shape of standard adult vowel-formant
    // charts (front vowels have high F2, back vowels low F2, open
    // vowels have high F1) -- typical/illustrative values, not a
    // calibrated measurement of any one voice.
    [PH_IY] = { 270, 2290, 3010,  60, 90, 150,  true, 1.00f, 150 }, // beet
    [PH_IH] = { 390, 1990, 2550,  65, 95, 150,  true, 1.00f, 110 }, // bit
    [PH_EH] = { 530, 1840, 2480,  70, 100, 150, true, 1.00f, 130 }, // bet
    [PH_AE] = { 660, 1720, 2410,  75, 100, 160, true, 1.00f, 160 }, // bat
    [PH_AA] = { 730, 1090, 2440,  80, 110, 160, true, 1.00f, 170 }, // father
    [PH_AO] = { 570, 840,  2410,  75, 100, 160, true, 1.00f, 160 }, // bought
    [PH_UH] = { 440, 1020, 2240,  65, 100, 150, true, 0.90f, 100 }, // book
    [PH_UW] = { 300, 870,  2240,  60, 95, 150,  true, 1.00f, 150 }, // boot
    [PH_AH] = { 640, 1190, 2390,  75, 100, 160, true, 1.00f, 110 }, // but
    [PH_AX] = { 500, 1500, 2500,  70, 110, 160, true, 0.60f, 70 },  // schwa

    // ---- Nasals ----
    // Characteristically low F1 and a damped, muffled quality (wider
    // bandwidths, reduced amplitude) relative to vowels.
    [PH_M]  = { 250, 1100, 2200,  60, 150, 200, true, 0.60f, 80 },
    [PH_N]  = { 250, 1600, 2400,  60, 150, 200, true, 0.60f, 80 },
    [PH_NG] = { 250, 2000, 2300,  60, 150, 200, true, 0.60f, 90 },

    // ---- Fricatives, unvoiced ----
    // Noise-sourced; the "formant" filters here shape the noise's
    // spectral peak(s) rather than modeling vocal-tract resonances.
    [PH_S]  = { 4500, 6000, 7000, 1000, 1000, 1000, false, 0.50f, 100 },
    [PH_F]  = { 2500, 4000, 6000, 1500, 1500, 1500, false, 0.35f, 90 },
    [PH_SH] = { 2200, 3000, 4000, 800, 800, 800,     false, 0.50f, 110 },
    [PH_TH] = { 3000, 5000, 7000, 1500, 1500, 1500,  false, 0.30f, 90 },
    [PH_H]  = { 1000, 2000, 3000, 1500, 1500, 1500,  false, 0.25f, 60 },

    // ---- Fricatives, voiced ----
    // Same spectral shaping as their unvoiced counterparts, but with a
    // pulsed (voiced) source instead of pure noise -- a simplification
    // of what's really a mixed periodic+noise excitation, giving a
    // "buzzy" rather than fully accurate voiced-fricative quality.
    [PH_Z]  = { 4500, 6000, 7000, 1000, 1000, 1000, true, 0.50f, 90 },
    [PH_V]  = { 2500, 4000, 6000, 1500, 1500, 1500, true, 0.40f, 80 },
    [PH_ZH] = { 2200, 3000, 4000, 800, 800, 800,    true, 0.50f, 100 },
    [PH_DH] = { 3000, 5000, 7000, 1500, 1500, 1500, true, 0.35f, 80 },

    // ---- Stops ----
    // Crude approximation: a single brief burst (noise for voiceless,
    // a short pulse for voiced), not a real closure-then-release model.
    [PH_P] = { 800, 1200, 2000,  1200, 1200, 1200, false, 0.40f, 40 },
    [PH_T] = { 3000, 4500, 6000, 1500, 1500, 1500, false, 0.40f, 40 },
    [PH_K] = { 1500, 2500, 3500, 1200, 1200, 1200, false, 0.40f, 45 },
    [PH_B] = { 400, 1000, 2200,  100, 150, 200,    true, 0.50f, 50 },
    [PH_D] = { 300, 1700, 2600,  100, 150, 200,    true, 0.50f, 50 },
    [PH_G] = { 250, 2000, 2700,  100, 150, 200,    true, 0.50f, 55 },

    // ---- Liquids / glides ----
    // Vowel-like: formant values approximate the closest associated
    // vowel quality. /r/'s characteristically low F3 is a real,
    // well-documented acoustic signature of American English /r/, not
    // an arbitrary choice.
    [PH_L] = { 360, 1300, 2700, 70, 110, 160, true, 0.80f, 90 },
    [PH_R] = { 310, 1060, 1380, 70, 110, 160, true, 0.80f, 90 }, // low F3 is the real signature of /r/
    [PH_W] = { 290, 610,  2150, 65, 95, 150,  true, 0.70f, 80 }, // like a UW onset
    [PH_Y] = { 280, 2200, 3000, 65, 95, 150,  true, 0.70f, 80 }, // like an IY onset
};