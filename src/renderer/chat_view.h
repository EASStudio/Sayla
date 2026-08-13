#pragma once

#define BUBBLE_FONT_SIZE   18
#define BUBBLE_PADDING     14.0f
#define BUBBLE_GAP         16.0f
#define BUBBLE_LINE_HEIGHT 24.0f
#define BUBBLE_MEASURE_BUF 600
#define BUBBLE_MAX_LINES   40
#define BUBBLE_SIDE_MARGIN 30.0f
#define AVATAR_GUTTER      40.0f
#define CHAT_VIEW_BOTTOM_RESERVED 220.0f
#define SCROLL_SPEED 60.0f
#define COPY_FEEDBACK_FRAMES 90   

// Library includes
#include "raylib.h"

void drawChatView(Rectangle area);