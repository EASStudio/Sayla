#pragma once

#define MAX_INPUT_CHARS 100000 
#define INPUT_BOX_HEIGHT 60.0f     
#define INPUT_BOX_MARGIN 40.0f  
#define INPUT_BOX_BOTTOM 30.0f   
#define BUTTON_RADIUS 20.0f
#define BUTTON_GAP 15.0f
#define BOX_ROUNDNESS 0.5f
#define BOX_SEGMENTS 16
#define LINE_HEIGHT 26.0f       
#define MAX_VISIBLE_LINES 6     
#define MAX_WRAP_LINES    64
#define LINE_MEASURE_BUF  600 
#define MAX_ATTACHMENTS   8
#define MAX_PATH_LEN      512
#define CHIP_HEIGHT       28.0f
#define CHIP_PAD          8.0f
#define CHIP_GAP          6.0f

// Library includes
#include "raylib.h"

typedef struct
{
    int start;  
    int length; 
} TextLine;

typedef struct
{
    char path[MAX_PATH_LEN];
    char name[128];          // display name only
    int  tokenCount;         // -1 if unknown (binary file, or unreadable)
} Attachment;

void drawInputBox(Rectangle area);
void sendUserMessage(const char *text);