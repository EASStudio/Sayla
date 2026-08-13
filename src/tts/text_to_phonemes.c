// Module includes
#include "text_to_phonemes.h"

// Core includes
#include <string.h>
#include <ctype.h>

typedef struct
{
    const char* pattern; // lowercase letters to match, matched as a whole unit
    Phoneme phonemes[3];
    int phonemeCount;
} G2PRule;

// Tried longest-pattern-first at each position (see textToPhonemes), so
// these don't need to be in any particular order relative to each other
// -- only exact-length matches are attempted at each tryLen.
static const G2PRule rules[] = {
    // 4-letter
    { "tion", { PH_SH, PH_AX, PH_N }, 3 },

    // 3-letter
    { "igh", { PH_AA, PH_IY }, 2 }, // "high", "light"
    { "tch", { PH_T, PH_SH }, 2 },
    { "dge", { PH_D, PH_ZH }, 2 },
    { "ing", { PH_IH, PH_NG }, 2 },

    // 2-letter
    { "th", { PH_TH }, 1 },
    { "sh", { PH_SH }, 1 },
    { "ch", { PH_T, PH_SH }, 2 },
    { "ph", { PH_F }, 1 },
    { "wh", { PH_W }, 1 },
    { "ng", { PH_NG }, 1 },
    { "ck", { PH_K }, 1 },
    { "qu", { PH_K, PH_W }, 2 },
    { "ee", { PH_IY }, 1 },
    { "ea", { PH_IY }, 1 },        // more often IY ("eat") than EH ("bread") -- picks the more common case
    { "oo", { PH_UW }, 1 },
    { "ou", { PH_AE, PH_UW }, 2 }, // "out", "house"
    { "ow", { PH_AE, PH_UW }, 2 }, // "now" -- same approximation as "ou"; "know" will mispronounce
    { "oi", { PH_AO, PH_IY }, 2 },
    { "oy", { PH_AO, PH_IY }, 2 },
    { "ai", { PH_EH, PH_IY }, 2 }, // "day", "rain"
    { "ay", { PH_EH, PH_IY }, 2 },
    { "oa", { PH_AO, PH_UW }, 2 }, // "boat", "road"
    { "ar", { PH_AA, PH_R }, 2 },
    { "er", { PH_AX, PH_R }, 2 },
    { "ir", { PH_AX, PH_R }, 2 },
    { "or", { PH_AO, PH_R }, 2 },
    { "ur", { PH_AX, PH_R }, 2 },

    // 1-letter fallbacks. This is where most of the remaining
    // inaccuracy lives -- English letters are genuinely ambiguous
    // without more context than this table tracks (e.g. "c"/"g"/"s"
    // each have two common sounds depending on the following letter,
    // not disambiguated here).
    { "a", { PH_AE }, 1 }, { "e", { PH_EH }, 1 }, { "i", { PH_IH }, 1 },
    { "o", { PH_AO }, 1 }, { "u", { PH_AH }, 1 }, { "y", { PH_IY }, 1 },
    { "b", { PH_B }, 1 },  { "c", { PH_K }, 1 },  { "d", { PH_D }, 1 },
    { "f", { PH_F }, 1 },  { "g", { PH_G }, 1 },  { "h", { PH_H }, 1 },
    { "j", { PH_D, PH_ZH }, 2 },
    { "k", { PH_K }, 1 },  { "l", { PH_L }, 1 },  { "m", { PH_M }, 1 },
    { "n", { PH_N }, 1 },  { "p", { PH_P }, 1 },  { "r", { PH_R }, 1 },
    { "s", { PH_S }, 1 },  { "t", { PH_T }, 1 },  { "v", { PH_V }, 1 },
    { "w", { PH_W }, 1 },
    { "x", { PH_K, PH_S }, 2 },
    { "z", { PH_Z }, 1 },
};

#define NUM_RULES (int)(sizeof(rules) / sizeof(rules[0]))
#define MAX_MATCH_LEN 4

int textToPhonemes(const char* text, Phoneme* outPhonemes, int maxPhonemes)
{
    int count = 0;
    int len = (int)strlen(text);
    int pos = 0;

    while (pos < len && count < maxPhonemes)
    {
        char c = text[pos];

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
        {
            outPhonemes[count++] = PH_SILENCE;
            pos++;
            continue;
        }

        if (c == '.' || c == '!' || c == '?' || c == ',' || c == ';' || c == ':')
        {
            // Actual punctuation -- a pause; sentence-enders get a
            // longer one.
            outPhonemes[count++] = PH_SILENCE;
            if ((c == '.' || c == '!' || c == '?') && count < maxPhonemes)
                outPhonemes[count++] = PH_SILENCE;
            pos++;
            continue;
        }

        if (!isalpha((unsigned char)c))
        {
            // Digits and anything else (not real punctuation, not a
            // letter) -- skip silently rather than either mangling it
            // into a wrong sound or, worse, treating it as a pause,
            // which would make any reply containing numbers go
            // strangely quiet. This does mean numbers aren't spoken at
            // all (no "5" -> "five" conversion) -- a real limitation,
            // not attempted here.
            pos++;
            continue;
        }

        bool matched = false;
        for (int tryLen = MAX_MATCH_LEN; tryLen >= 1 && !matched; tryLen--)
        {
            if (pos + tryLen > len) continue;

            char buf[MAX_MATCH_LEN + 1];
            for (int i = 0; i < tryLen; i++) buf[i] = (char)tolower((unsigned char)text[pos + i]);
            buf[tryLen] = '\0';

            for (int r = 0; r < NUM_RULES; r++)
            {
                if ((int)strlen(rules[r].pattern) == tryLen && strcmp(rules[r].pattern, buf) == 0)
                {
                    for (int k = 0; k < rules[r].phonemeCount && count < maxPhonemes; k++)
                        outPhonemes[count++] = rules[r].phonemes[k];
                    pos += tryLen;
                    matched = true;
                    break;
                }
            }
        }

        if (!matched)
            pos++; // unhandled character (digit, symbol) -- skip rather than emit garbage
    }

    return count;
}