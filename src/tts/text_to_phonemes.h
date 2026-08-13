#pragma once

#include "phonemes.h"

// Converts English text into a phoneme sequence using hand-written
// spelling-to-sound rules (longest-pattern-first matching) -- not a
// pronunciation dictionary and not a trained model. English spelling is
// full of context-dependent exceptions this kind of simple rule table
// cannot capture (the letter "c" sounds different in "cat" vs "city";
// "read" is pronounced differently in "I read" vs "I will read"), so
// expect real mispronunciations, especially on irregular words.
// Diphthongs (the vowel glide in "day", "boy", "now", etc.) are
// approximated as a short two-phoneme sequence from the existing vowel
// set rather than adding dedicated diphthong phonemes. Digits and any
// other non-letter, non-punctuation characters are silently skipped
// (no "5" -> "five" conversion) rather than mangled or read as pauses.


int textToPhonemes(const char* text, Phoneme* outPhonemes, int maxPhonemes);