// Module includes
#include "voice.h"
#include "raylib.h" 
#include "../tts/text_to_phonemes.h"
#include "../tts/format_synth.h"

// Core includes
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// ===========================================================================
// Speaking: from-scratch formant synthesis (see ../tts/). No external
// library, so this is NOT gated behind VOICE_ENABLED -- it's always
// available. speak() and updateVoice()'s playback pump both run on the
// main thread only (nothing here has its own worker thread the way
// espeak's callback used to), so none of this needs the mutex the
// listening side below uses.
// ===========================================================================

static AudioStream ttsStream;
static bool ttsStreamReady = false;

static short ttsBuffer[TTS_BUFFER_CAPACITY];
static int ttsWritePos = 0; // end of rendered audio for the current reply
static int ttsReadPos = 0;  // next index handed to raylib's AudioStream
static bool speaking = false;
static float speakingLevel = 0.0f;

static void initSpeaking(void)
{
    // Must be called BEFORE LoadAudioStream() -- it sets the default
    // sub-buffer size used when a new stream is created (this is
    // raylib's own documented pattern, e.g. its audio_raw_stream.c
    // example). Without this, raylib's own internal default (smaller
    // than the 4096-sample chunks updateSpeaking() below feeds it)
    // caused a flood of "Attempting to write too many frames to
    // buffer" warnings on every single feed -- hundreds of console
    // writes per second, which is slow enough to badly stutter the
    // main thread. That thread is exactly where transcription handling
    // and speak() run, so this alone was a plausible root cause of
    // playback looking desynchronized from what was actually said,
    // separate from (and on top of) the polling race fixed earlier.
    SetAudioStreamBufferSizeDefault(8192); // comfortably above the 4096-sample chunk cap below
    ttsStream = LoadAudioStream(TTS_SAMPLE_RATE, 16, 1);
    ttsStreamReady = true;
}

static void freeSpeaking(void)
{
    if (ttsStreamReady)
    {
        UnloadAudioStream(ttsStream);
        ttsStreamReady = false;
    }
}

static void updateSpeaking(void)
{
    if (!ttsStreamReady) return;

    // A while loop, not a single if: raylib's own source confirms
    // AudioStream uses two internal sub-buffers, and
    // IsAudioStreamProcessed() reports true if EITHER ONE needs
    // refilling ("isSubBufferProcessed[0] || isSubBufferProcessed[1]").
    // A single "if" only ever refills one sub-buffer per call to this
    // function -- if both empty out in the same frame (plausible after
    // even a brief stall elsewhere), the second one sits empty until
    // the NEXT frame's check, which is an audible gap in playback, not
    // just a missed optimization. This matters here specifically:
    // reported symptom was audio "breaking up" repeatedly through a
    // reply, which is what recurring sub-buffer starvation sounds like.
    // Looping keeps feeding for as long as a sub-buffer still needs it
    // and there's data left to give it; it terminates on its own once
    // either condition goes false, and since raylib only ever has two
    // sub-buffers to refill, this can run at most a couple of times
    // per call even after a long stall, not runaway.
    while (ttsWritePos - ttsReadPos > 0 && IsAudioStreamProcessed(ttsStream))
    {
        int available = ttsWritePos - ttsReadPos;
        int chunk = available;
        if (chunk > 8192) chunk = 8192; // matches SetAudioStreamBufferSizeDefault(8192) in initSpeaking() -- more buffered playback time per feed, more margin before starvation

        // Real amplitude of the chunk actually being handed to playback
        // right now, not an average over the whole reply -- keeps the
        // ring honestly synced to what's actually about to be heard.
        double sumSquares = 0.0;
        for (int i = 0; i < chunk; i++)
        {
            double s = (double)ttsBuffer[ttsReadPos + i] / 32768.0;
            sumSquares += s * s;
        }
        float rms = (float)sqrt(sumSquares / chunk);
        speakingLevel = rms * 3.0f;
        if (speakingLevel > 1.0f) speakingLevel = 1.0f;

        UpdateAudioStream(ttsStream, &ttsBuffer[ttsReadPos], chunk);
        ttsReadPos += chunk;

        if (!IsAudioStreamPlaying(ttsStream))
        {
            // DIAGNOSTIC (temporary): if speak() logged a nonzero
            // sample count above but this never prints, the rendered
            // audio never actually started playing through raylib --
            // that would point at the AudioStream/audio-device side
            // specifically, not the synthesizer.
            printf("[voice] starting audio playback\n");
            PlayAudioStream(ttsStream);
        }
    }

    bool stillQueued = (ttsReadPos < ttsWritePos);
    speaking = stillQueued || IsAudioStreamPlaying(ttsStream);

    if (!speaking) speakingLevel = 0.0f;
}

void speak(const char* text)
{
    if (!ttsStreamReady || text == NULL || text[0] == '\0')
    {
        // DIAGNOSTIC (temporary): if this prints ttsStreamReady=0, the
        // AudioStream was never successfully created -- check for an
        // "AUDIO: Failed to initialize" line in the console right after
        // startup, since that would mean InitAudioDevice() itself
        // failed (a real, if unusual, hardware/driver issue), not
        // anything in this file.
        printf("[voice] speak() early-return: ttsStreamReady=%d, text=%s\n",
               ttsStreamReady, text ? "(non-null)" : "(NULL)");
        return;
    }

    static Phoneme phonemes[MAX_PHONEMES];
    int phonemeCount = textToPhonemes(text, phonemes, MAX_PHONEMES);

    // Synchronous, not threaded -- measured over 500x faster than
    // real-time even for a long reply (see formant_synth's own test),
    // so this can't cause a perceptible hitch the way a neural TTS call
    // could.
    int rendered = synthesizeSpeech(phonemes, phonemeCount, ttsBuffer, TTS_BUFFER_CAPACITY, TTS_SAMPLE_RATE);

    // DIAGNOSTIC (temporary): confirms speak() was actually called and
    // shows exactly what got synthesized. If phonemeCount is 0, the text
    // itself didn't map to anything speakable (check what's actually
    // being passed in). If rendered is 0 despite a nonzero phonemeCount,
    // that's a real bug in formant_synth.c worth reporting back.
    printf("[voice] speak(\"%s\") -> %d phonemes, %d samples rendered (%.2fs of audio)\n",
           text, phonemeCount, rendered, (double)rendered / TTS_SAMPLE_RATE);

    ttsWritePos = rendered;
    ttsReadPos = 0;
    speaking = (rendered > 0);
}

bool isSpeaking(void) { return speaking; }
float getSpeakingLevel(void) { return speakingLevel; }

// ===========================================================================
// Listening: whisper.cpp (speech-to-text) + miniaudio (mic capture) +
// this project's own VAD (see ../vad/). Gated behind VOICE_ENABLED --
// unlike speaking, this genuinely needs an external library
// (whisper.cpp) for open-vocabulary dictation.
// ===========================================================================

#ifdef VOICE_ENABLED

#include "whisper.h"
#include "miniaudio.h"
#include "vad.h"

// ---- Portable mutex/thread shim ----------------------------------------
// Identical to the fix already proven in engine.c: SRWLOCK (not
// CRITICAL_SECTION) on Windows, since SRWLOCK_INIT is officially safe to
// zero-initialize statically and CRITICAL_SECTION is not.
//
// This file (unlike engine.c) also includes raylib.h above, for
// AudioStream -- and raylib.h and <windows.h> define three of the same
// names (Rectangle, CloseWindow, ShowCursor). NOGDI/NOUSER strip
// exactly the GDI/User32 declarations responsible for those three names
// out of <windows.h>; nothing this shim actually uses (SRWLOCK,
// _beginthreadex, WaitForSingleObject, CloseHandle) lives in GDI or
// User32, so this removes nothing needed.
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOGDI
  #define NOUSER
  #include <windows.h>
  #include <process.h>

  #define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
  #define pthread_mutex_t SRWLOCK
  #define pthread_mutex_lock(mtx) (AcquireSRWLockExclusive(mtx), 0)
  #define pthread_mutex_unlock(mtx) ReleaseSRWLockExclusive(mtx)
  #define pthread_t HANDLE

  #define pthread_create(thread, attr, start_routine, arg) \
    (((*(thread) = (HANDLE)_beginthreadex( \
        NULL, 0, \
        (unsigned (__stdcall *)(void *))(start_routine), \
        (arg), 0, NULL)) == (HANDLE)-1) ? 1 : 0)

  #define pthread_join(thread, retval) \
    ((WaitForSingleObject((thread), INFINITE) == WAIT_OBJECT_0) ? \
    (CloseHandle(thread), 0) : -1)
#endif

#ifndef _WIN32
  #include <pthread.h>
#endif

#define CAPTURE_MAX_SAMPLES (30 * WHISPER_SAMPLE_RATE) // 30s cap on one VAD-detected utterance

static struct whisper_context* whisperCtx = NULL;

static ma_device captureDevice;
static bool captureDeviceReady = false;

// Guards every piece of state below (touched from the miniaudio capture
// callback's own thread, the background transcription thread, and the
// main thread). A plain mutex inside an audio callback isn't what a
// professional low-latency audio engine would do, but for a personal
// project this is simple and correct, which matters more here.
static pthread_mutex_t voiceMutex = PTHREAD_MUTEX_INITIALIZER;

static float captureBuffer[CAPTURE_MAX_SAMPLES];
static int captureSampleCount = 0;
static bool liveListeningEnabled = false;
static bool userSpeaking = false;
static bool utterancePendingTranscription = false;
static float micLevel = 0.0f;

static pthread_t transcribeThread;
static bool transcribing = false;
static bool transcriptionReady = false;
static char transcriptionText[4096];

// Fires on miniaudio's own internal audio thread, not the main thread.
// Does only cheap, bounded work -- deliberately does NOT create the
// transcription thread here; that happens from updateVoice() on the
// main thread instead (see utterancePendingTranscription), since
// spawning a thread from inside a real-time audio callback risks a
// glitch.
static void captureCallback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    (void)pDevice; (void)pOutput;
    const float* in = (const float*)pInput;

    // DIAGNOSTIC (temporary): prints exactly once, the first time
    // miniaudio actually delivers a buffer of mic audio to this
    // callback at all. If "capture device init: OK" printed but this
    // NEVER prints while listening is on, miniaudio opened a device
    // handle but isn't actually receiving audio from it -- often a
    // wrong/silent default input device, or a Windows privacy setting
    // blocking this specific app from the mic even though the device
    // itself opened without error.
    static bool loggedFirstCallback = false;
    if (!loggedFirstCallback)
    {
        printf("[voice] capture callback firing (frameCount=%u) -- miniaudio is delivering audio\n", frameCount);
        loggedFirstCallback = true;
    }

    pthread_mutex_lock(&voiceMutex);

    VadEvent event = vadProcess(in, (int)frameCount);

    // Use VAD's own level instead of a separately-computed raw RMS --
    // vadGetLevel() is already smoothed (an exponential moving average,
    // not a raw instantaneous value that would look jittery frame to
    // frame) and already normalized against the adaptive noise floor,
    // so "1.0" means "clearly above this room/mic's own ambient level"
    // rather than an arbitrary fixed gain multiplier guessed blind
    // without real hardware to test against. Calling vadGetLevel() here
    // (not from getMicLevel(), which the main thread calls) matters:
    // vad.c's internals aren't separately mutex-protected, so reading
    // them is only safe from the same thread that calls vadProcess().
    micLevel = vadGetLevel();

    // DIAGNOSTIC (temporary): VAD events are naturally rare (once per
    // utterance boundary), so printing every one isn't spammy. If
    // captureCallback is confirmed firing above but SPEECH_STARTED
    // never prints no matter how loud/close you talk, the mic's real
    // input level is likely far lower than the synthetic signals VAD
    // was tuned against -- worth trying cranking your OS input volume
    // or moving closer to the mic before assuming it's a code bug.
    if (event == VAD_EVENT_SPEECH_STARTED) printf("[voice] VAD: speech started\n");
    if (event == VAD_EVENT_SPEECH_ENDED) printf("[voice] VAD: speech ended, queuing for transcription\n");

    // Ignore a new speech-started while a previous utterance is still
    // being transcribed -- captureBuffer is single-buffered.
    if (event == VAD_EVENT_SPEECH_STARTED && !transcribing)
    {
        captureSampleCount = 0;
        userSpeaking = true;
    }

    if (userSpeaking)
    {
        for (ma_uint32 i = 0; i < frameCount; i++)
            if (captureSampleCount < CAPTURE_MAX_SAMPLES)
                captureBuffer[captureSampleCount++] = in[i];
    }

    if (event == VAD_EVENT_SPEECH_ENDED && userSpeaking)
    {
        userSpeaking = false;
        utterancePendingTranscription = true;
    }

    pthread_mutex_unlock(&voiceMutex);
}

// Runs whisper_full() off the main thread so a long utterance doesn't
// freeze the UI for however long inference takes.
static void* transcribeThreadFunc(void* arg)
{
    (void)arg;

    static float localBuffer[CAPTURE_MAX_SAMPLES]; // static: avoid a ~2MB stack frame
    int n;

    pthread_mutex_lock(&voiceMutex);
    n = captureSampleCount;
    memcpy(localBuffer, captureBuffer, sizeof(float) * (size_t)n);
    pthread_mutex_unlock(&voiceMutex);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = 4;
    params.translate = false;
    params.no_context = true;
    params.no_timestamps = true;
    params.single_segment = true;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.language = "en";
    params.detect_language = false;

    char result[4096];
    result[0] = '\0';

    // DIAGNOSTIC (temporary): if "VAD: speech ended" printed but this
    // shows n=0 samples, the capture buffer was empty despite VAD
    // triggering (a real bug worth reporting back, not an environment
    // issue). If n is reasonable but whisper_full fails or the
    // transcribed text comes back empty, the model loaded but couldn't
    // make sense of the audio -- check the mic is actually the input
    // device you expect (some systems default to a virtual/silent
    // device), or try a larger whisper model.
    printf("[voice] transcribing %d captured samples (%.1fs of audio)\n", n, (double)n / WHISPER_SAMPLE_RATE);

    if (whisperCtx != NULL && n > 0 && whisper_full(whisperCtx, params, localBuffer, n) == 0)
    {
        int segments = whisper_full_n_segments(whisperCtx);
        for (int i = 0; i < segments; i++)
        {
            const char* text = whisper_full_get_segment_text(whisperCtx, i);
            if (text != NULL)
                strncat(result, text, sizeof(result) - strlen(result) - 1);
        }
    }

    printf("[voice] transcription result: \"%s\"\n", result);

    pthread_mutex_lock(&voiceMutex);
    strncpy(transcriptionText, result, sizeof(transcriptionText) - 1);
    transcriptionText[sizeof(transcriptionText) - 1] = '\0';
    transcriptionReady = true;
    transcribing = false;
    pthread_mutex_unlock(&voiceMutex);

    return NULL;
}

bool initVoice(const char* whisperModelPath)
{
    initSpeaking();

    struct whisper_context_params cparams = whisper_context_default_params();
    whisperCtx = whisper_init_from_file_with_params(whisperModelPath, cparams);
    if (whisperCtx == NULL)
    {
        // DIAGNOSTIC (temporary): if this prints, the ggml model file
        // at whisperModelPath either doesn't exist or failed to parse
        // -- listening will silently stay off with no other symptom.
        // Check the path is correct relative to wherever you're
        // actually running the .exe from (not necessarily your project
        // root), and that the file downloaded completely.
        printf("[voice] whisper_init_from_file_with_params FAILED for path: %s\n", whisperModelPath);
        return false; // listening unavailable; speaking still works regardless
    }
    printf("[voice] whisper model loaded OK: %s\n", whisperModelPath);

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = WHISPER_SAMPLE_RATE;
    config.dataCallback = captureCallback;
    config.pUserData = NULL;

    captureDeviceReady = (ma_device_init(NULL, &config, &captureDevice) == MA_SUCCESS);
    // DIAGNOSTIC (temporary): if this prints false, miniaudio couldn't
    // open a capture device at all -- likely no microphone available,
    // a permissions issue (Windows mic privacy settings can block
    // this per-app), or another application holding it exclusively.
    printf("[voice] capture device init: %s\n", captureDeviceReady ? "OK" : "FAILED");

    initVAD(WHISPER_SAMPLE_RATE);

    return captureDeviceReady;
}

void freeVoice(void)
{
    if (liveListeningEnabled)
        stopLiveListening();

    if (transcribing)
        pthread_join(transcribeThread, NULL);

    if (captureDeviceReady)
    {
        ma_device_uninit(&captureDevice);
        captureDeviceReady = false;
    }

    if (whisperCtx != NULL)
    {
        whisper_free(whisperCtx);
        whisperCtx = NULL;
    }

    freeSpeaking();
}

void updateVoice(void)
{
    updateSpeaking();

    // VAD -> transcription handoff. Thread creation deliberately
    // happens here (main thread), not inside captureCallback().
    pthread_mutex_lock(&voiceMutex);
    bool pending = utterancePendingTranscription;
    if (pending) utterancePendingTranscription = false;
    pthread_mutex_unlock(&voiceMutex);

    if (pending && !transcribing)
    {
        pthread_mutex_lock(&voiceMutex);
        transcribing = true;
        transcriptionReady = false;
        pthread_mutex_unlock(&voiceMutex);

        pthread_create(&transcribeThread, NULL, transcribeThreadFunc, NULL);
    }
}

void startLiveListening(void)
{
    if (!captureDeviceReady || liveListeningEnabled) return;

    pthread_mutex_lock(&voiceMutex);
    captureSampleCount = 0;
    userSpeaking = false;
    utterancePendingTranscription = false;
    micLevel = 0.0f;
    pthread_mutex_unlock(&voiceMutex);

    resetVAD();
    ma_device_start(&captureDevice);
    liveListeningEnabled = true;
}

void stopLiveListening(void)
{
    if (!liveListeningEnabled) return;

    ma_device_stop(&captureDevice);
    liveListeningEnabled = false;

    pthread_mutex_lock(&voiceMutex);
    // Discard any partial utterance -- the user turned listening off
    // mid-sentence, so transcribing a truncated fragment isn't useful.
    userSpeaking = false;
    utterancePendingTranscription = false;
    micLevel = 0.0f;
    pthread_mutex_unlock(&voiceMutex);
}

bool isLiveListeningEnabled(void) { return liveListeningEnabled; }
bool isUserSpeaking(void) { return userSpeaking; }
bool isTranscribing(void) { return transcribing; }

bool pollTranscription(char* outText, int maxLen)
{
    bool result = false;

    pthread_mutex_lock(&voiceMutex);
    if (transcriptionReady)
    {
        strncpy(outText, transcriptionText, (size_t)maxLen - 1);
        outText[maxLen - 1] = '\0';
        transcriptionReady = false;
        result = true;
    }
    pthread_mutex_unlock(&voiceMutex);

    return result;
}

float getMicLevel(void)
{
    pthread_mutex_lock(&voiceMutex);
    float level = micLevel;
    pthread_mutex_unlock(&voiceMutex);
    return level;
}

#else // !VOICE_ENABLED -- speaking above still works fully; only
      // listening degrades to harmless no-ops/false so the rest of the
      // app (including the Live panel UI) builds and runs fine without
      // whisper.cpp wired in.

bool initVoice(const char* whisperModelPath)
{
    (void)whisperModelPath;
    initSpeaking();
    return false; // listening unavailable; speaking still works
}

void freeVoice(void) { freeSpeaking(); }
void updateVoice(void) { updateSpeaking(); }
void startLiveListening(void) {}
void stopLiveListening(void) {}
bool isLiveListeningEnabled(void) { return false; }
bool isUserSpeaking(void) { return false; }
bool isTranscribing(void) { return false; }
bool pollTranscription(char* outText, int maxLen) { (void)outText; (void)maxLen; return false; }
float getMicLevel(void) { return 0.0f; }

#endif