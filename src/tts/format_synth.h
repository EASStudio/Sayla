#pragma once

#include "phonemes.h"

#define BASE_F0_START 130.0f // Hz, pitch at the start of an utterance
#define BASE_F0_END   95.0f  // Hz, pitch at the end -- a simple declining

#define FORMANT_TRANSITION_MS 40.0f
#define ATTACK_FRACTION 0.15f
#define DECAY_FRACTION 0.20f

// Relative weight of each formant in the final mix -- F1 and F2 carry
// most of a vowel's perceptual identity, F3 contributes less; weighting
// it down some avoids it dominating and sounding harsh.
#define F1_WEIGHT 1.0f
#define F2_WEIGHT 0.85f
#define F3_WEIGHT 0.55f

// Overall output gain, tuned empirically against a range of phonemes
// (see this module's accompanying test) so typical output sits
// comfortably below 16-bit clipping without being inaudibly quiet.
#define OUTPUT_GAIN 64000.0f

// Renders a phoneme sequence to 16-bit mono PCM, from scratch: a
// pulsed source (voiced sounds) or noise source (unvoiced sounds) fed
// into three resonant filters in parallel (see resonator.h), one per
// formant, summed together. Formant targets glide smoothly from one
// phoneme to the next rather than jumping instantly. Pitch follows a
// simple declining contour over the whole utterance for basic
// declarative-sentence intonation -- no per-word stress, no question
// rise, no real prosody model.

int synthesizeSpeech(const Phoneme* phonemes, int phonemeCount, short* outSamples, int maxSamples, int sampleRate);