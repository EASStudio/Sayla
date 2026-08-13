#pragma once

#include <stdbool.h>

#define TTS_SAMPLE_RATE 22050
#define TTS_BUFFER_CAPACITY 2000000 // generous cap on one spoken reply
#define MAX_PHONEMES 4096

bool initVoice(const char* whisperModelPath);
void freeVoice(void);
void updateVoice(void);
void startLiveListening(void);
void stopLiveListening(void);
bool isLiveListeningEnabled(void);
bool isUserSpeaking(void);
bool isTranscribing(void);
bool pollTranscription(char* outText, int maxLen);
float getMicLevel(void);
void speak(const char* text);
bool isSpeaking(void);
float getSpeakingLevel(void);