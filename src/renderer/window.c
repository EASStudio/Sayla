// Module includes
#include "window.h"
#include "welcome.h"
#include "sidebar.h"
#include "chat_view.h"
#include "thinking.h"
#include "live_view.h"
#include "../llm/input.h"
#include "../llm/chat_history.h"
#include "../llm/engine.h"
#include "../vad/voice.h"

// Core includes
#include <stdio.h>

void createWindow()
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    SetConfigFlags(FLAG_WINDOW_HIGHDPI);

    InitWindow(INITIAL_WIDTH, INITIAL_HEIGHT, TITLE);
    SetWindowMinSize(MIN_WIDTH, MIN_HEIGHT);

    SetTargetFPS(60);

    InitAudioDevice();

    initChatHistory();

    initVoice(WHISPER_MODEL_PATH);

    startEngineInitAsync();

    while (!WindowShouldClose())
    {
        int screenWidth  = GetScreenWidth();
        int screenHeight = GetScreenHeight();

        TrainingProgress progress = getTrainingProgress();

        updateVoice();

        BeginDrawing();
        ClearBackground(RAYWHITE);

        if (!progress.isDone)
        {
            char label[160];
            if (progress.isTraining && progress.totalSteps > 0)
            {
                snprintf(label, sizeof(label), "Training %s: step %ld / %ld (loss %.3f)",
                         progress.domainName, progress.step, progress.totalSteps, progress.lastLoss);
            }
            
            else
            {
                snprintf(label, sizeof(label), "Preparing %s...", progress.domainName[0] != '\0' ? progress.domainName : "models");
            }

            Rectangle fullScreen = { 0, 0, (float)screenWidth, (float)screenHeight };
            drawCenteredThinkingScreen(fullScreen, label);

            EndDrawing();
            continue;
        }

        Rectangle contentArea =
        {
            SIDEBAR_WIDTH,
            0,
            (float)screenWidth - SIDEBAR_WIDTH,
            (float)screenHeight
        };

        // Ctrl+S to save
        bool ctrlHeld = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL) || IsKeyDown(KEY_LEFT_SUPER)   || IsKeyDown(KEY_RIGHT_SUPER);
        if (ctrlHeld && IsKeyPressed(KEY_S))
        {
            saveChatHistoryToFile(CHAT_HISTORY_SAVE_PATH);
        }

        drawSidebar();

        if (getSidebarPanelMode() == PANEL_LIVE)
        {
            drawLivePanel(contentArea);
        }
        else
        {
            if (getChatMessageCount() == 0 && !isThinking())
            {
                drawWelcomeBar(contentArea);
            }
            else
            {
                drawChatView(contentArea);
            }

            drawInputBox(contentArea);
        }

        EndDrawing();
    }

    freeVoice();
    freeEngine();
    CloseAudioDevice();
    CloseWindow();
}