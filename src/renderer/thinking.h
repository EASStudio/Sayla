#pragma once

// Library includes
#include "raylib.h"

// Core includes
#include <stdbool.h>

void drawAnimatedAvatar(Vector2 center, float radius);
void drawCenteredThinkingScreen(Rectangle area, const char* label);
void beginThinking(void);
bool isThinkingDone(void);
void endThinking(void);
bool isThinking(void);
const char* currentThinkingLabel(void);