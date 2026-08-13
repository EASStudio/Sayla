#pragma once

// Library includes
#include "raylib.h"

#define INITIAL_WIDTH  1080
#define INITIAL_HEIGHT 720
#define MIN_WIDTH 760
#define MIN_HEIGHT 480
#define TITLE "Sayla"
#define CHAT_HISTORY_SAVE_PATH "chat_history.txt"
#define WHISPER_MODEL_PATH "models/ggml-base.en.bin"

void createWindow(void);