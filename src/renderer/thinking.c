// Module includes
#include "thinking.h"
#include "avatar.h"
#include "colors.h"

// Core includes
#include <math.h>
#include <stddef.h>

void drawAnimatedAvatar(Vector2 center, float radius)
{
    float t = (float)GetTime();
    float bob = sinf(t * 2.0f) * radius * 0.06f;
    float flutter = 1.0f + sinf(t * 6.0f) * 0.03f;

    Vector2 animatedCenter = { center.x, center.y + bob };
    drawBirdAvatar(animatedCenter, radius * flutter);
}

void drawCenteredThinkingScreen(Rectangle area, const char* label)
{
    Vector2 center = { area.x + area.width / 2.0f, area.y + area.height / 2.0f - 40.0f };
    drawAnimatedAvatar(center, 46.0f);

    int fontSize = 18;
    int textWidth = MeasureText(label, fontSize);
    DrawText(label, (int)(center.x - textWidth / 2.0f), (int)(center.y + 70.0f), fontSize, COLOR_WELCOME_TEXT);
}

// ---- Rotating status labels ---------------------------------------------
static const char* thinkingLabels[] = {
    "Reading your message...",
    "Choosing the right specialist...",
    "Working through a response...",
};
#define NUM_THINKING_LABELS ((int)(sizeof(thinkingLabels) / sizeof(thinkingLabels[0])))

const char* currentThinkingLabel(void)
{
    int index = ((int)(GetTime() * 3.0)) % NUM_THINKING_LABELS;
    return thinkingLabels[index];
}

// ---- Inline thinking beat -------------------------------------------
#define THINKING_MIN_DURATION 0.7 // seconds -- long enough to read one label, short enough not to feel slow

static bool thinkingActive = false;
static double thinkingStartTime = 0.0;

void beginThinking(void)
{
    thinkingActive = true;
    thinkingStartTime = GetTime();
}

bool isThinkingDone(void)
{
    if (!thinkingActive) return true;
    return (GetTime() - thinkingStartTime) >= THINKING_MIN_DURATION;
}

void endThinking(void)
{
    thinkingActive = false;
}

bool isThinking(void)
{
    return thinkingActive;
}