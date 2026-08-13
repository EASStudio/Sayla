// Module includes
#include "live_view.h"
#include "colors.h"
#include "thinking.h"          // drawAnimatedAvatar
#include "../llm/chat_history.h"
#include "../llm/engine.h"
#include "../vad/voice.h"      // sendUserMessage-triggered speak(), listening state
#include "../llm/input.h"      // sendUserMessage

// Core includes
#include <string.h>
#include <stdio.h>
#include <math.h>

// Draws 1-3 concentric rings around the avatar, sized/faded by the given
// audio level (0..1, from voice.c's real mic/TTS amplitude -- see
// drawLivePanel below for which one feeds this at any given moment).
static void drawAudioRing(Vector2 center, float baseRadius, float level, Color ringColor)
{
    for (int ring = 0; ring < 3; ring++)
    {
        float spread = 1.0f + ring * 0.15f + level * 0.5f;
        float alpha = (0.40f - ring * 0.10f) * (0.35f + level * 1.3f);
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;
        DrawCircleLines((int)center.x, (int)center.y, baseRadius * spread, Fade(ringColor, alpha));
    }
}

// Smooths whichever raw level is driving the ring this frame (mic,
// speaking, or the idle pulse) so its motion reads as fluid rather than
// jittery -- any per-frame raw value benefits from this regardless of
// source, since real audio's frame-to-frame amplitude is naturally
// noisy even when VAD/TTS are working correctly.
static float smoothRingLevel(float rawLevel)
{
    static float smoothed = 0.0f;
    smoothed = smoothed * 0.75f + rawLevel * 0.25f;
    return smoothed;
}

static void drawMicButton(Vector2 center, float radius, bool active)
{
    Color bg = active ? Fade(COLOR_SEND_ON, 0.85f) : COLOR_UPLOAD_BG;
    DrawCircleV(center, radius, bg);
    DrawCircleLines((int)center.x, (int)center.y, radius, Fade(COLOR_UPLOAD_ICON, 0.4f));

    // Simple mic glyph: a small rounded body + a base line -- good enough
    // at this size without needing a texture or icon font.
    Rectangle body = { center.x - radius * 0.22f, center.y - radius * 0.45f, radius * 0.44f, radius * 0.65f };
    Color glyph = active ? WHITE : COLOR_UPLOAD_ICON;
    DrawRectangleRounded(body, 0.9f, 8, glyph);
    DrawLineEx((Vector2){ center.x, center.y + radius * 0.25f }, (Vector2){ center.x, center.y + radius * 0.45f }, 2.0f, glyph);
    DrawLineEx((Vector2){ center.x - radius * 0.22f, center.y + radius * 0.45f },
               (Vector2){ center.x + radius * 0.22f, center.y + radius * 0.45f }, 2.0f, glyph);
}

// Shows the most recent assistant reply as a simple text panel, standing
// in for a real "program files" view. The engine currently returns one
// flat text response per turn rather than structured, named files, so
// this is the most honest thing to show without inventing file data that
// doesn't exist -- a real files panel would need the engine to track
// generated files separately, a bigger change than this pass.
#define MAX_CODE_FILES 8
#define CODE_FILE_MARKER "// file: "

typedef struct
{
    char filename[64]; // points into the response text -- not a copy, not independently null-terminated
    const char* contentStart;
    int contentLen;
} CodeFileRef;

// Finds the start index of every line that begins with CODE_FILE_MARKER
// -- a marker string appearing mid-line (e.g. inside a printf call) is
// deliberately not treated as a real marker.
static int findMarkers(const char* text, int len, int* outPositions, int maxMarkers)
{
    int markerLen = (int)strlen(CODE_FILE_MARKER);
    int count = 0;
    for (int i = 0; i < len && count < maxMarkers; i++)
    {
        bool atLineStart = (i == 0) || (text[i - 1] == '\n');
        if (atLineStart && (i + markerLen <= len) && strncmp(&text[i], CODE_FILE_MARKER, markerLen) == 0)
            outPositions[count++] = i;
    }
    return count;
}

// Splits response text into 1+ named files using a "// file: name.ext"
// line-start marker convention. Verified against a standalone test
// (single file, multi-file, preamble handling, cap respecting, and
// rejecting a marker-like string that isn't actually at a line start)
// before being wired in here. If no markers are found -- the common
// case today, since the training corpus doesn't yet produce this format
// -- returns exactly one file named "response" holding the whole text
// unchanged. This is real, working multi-file *display* infrastructure;
// it does not make the underlying model smarter about deciding when a
// task genuinely needs multiple files.
static int splitIntoCodeFiles(const char* text, CodeFileRef* outFiles, int maxFiles)
{
    int len = (int)strlen(text);
    int markerLen = (int)strlen(CODE_FILE_MARKER);

    int markerPositions[MAX_CODE_FILES];
    int markerCount = findMarkers(text, len, markerPositions, MAX_CODE_FILES);

    if (markerCount == 0)
    {
        if (maxFiles <= 0) return 0;
        strncpy(outFiles[0].filename, "response", sizeof(outFiles[0].filename) - 1);
        outFiles[0].filename[sizeof(outFiles[0].filename) - 1] = '\0';
        outFiles[0].contentStart = text;
        outFiles[0].contentLen = len;
        return 1;
    }

    int fileCount = 0;
    for (int m = 0; m < markerCount && fileCount < maxFiles; m++)
    {
        int nameStart = markerPositions[m] + markerLen;
        int nameEnd = nameStart;
        while (nameEnd < len && text[nameEnd] != '\n') nameEnd++;

        int nameLen = nameEnd - nameStart;
        if (nameLen > 63) nameLen = 63;
        memcpy(outFiles[fileCount].filename, &text[nameStart], nameLen);
        outFiles[fileCount].filename[nameLen] = '\0';

        int contentStart = (nameEnd < len) ? nameEnd + 1 : nameEnd;
        int contentEnd = (m + 1 < markerCount) ? markerPositions[m + 1] : len;
        if (contentEnd < contentStart) contentEnd = contentStart;

        outFiles[fileCount].contentStart = &text[contentStart];
        outFiles[fileCount].contentLen = contentEnd - contentStart;
        fileCount++;
    }

    return fileCount;
}

// A small clickable "Copy" button that puts the given file's content on
// the system clipboard, with brief "Copied!" visual feedback.
static void drawCopyButton(Rectangle area, const CodeFileRef* file)
{
    static float copyFeedbackTimer = 0.0f;
    copyFeedbackTimer -= GetFrameTime();
    if (copyFeedbackTimer < 0.0f) copyFeedbackTimer = 0.0f;

    Vector2 mouse = GetMousePosition();
    bool hovering = CheckCollisionPointRec(mouse, area);

    Color bg = hovering ? Fade(COLOR_SEND_ON, 0.85f) : COLOR_UPLOAD_BG;
    DrawRectangleRounded(area, 0.3f, 6, bg);

    const char* label = (copyFeedbackTimer > 0.0f) ? "Copied!" : "Copy";
    int fontSize = 13;
    int tw = MeasureText(label, fontSize);
    DrawText(label, (int)(area.x + (area.width - tw) / 2.0f), (int)(area.y + (area.height - fontSize) / 2.0f),
             fontSize, hovering ? WHITE : COLOR_UPLOAD_ICON);

    if (hovering)
        SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);

    if (hovering && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
    {
        static char clipboardScratch[MAX_MESSAGE_LENGTH];
        int n = file->contentLen;
        if (n > (int)sizeof(clipboardScratch) - 1) n = (int)sizeof(clipboardScratch) - 1;
        memcpy(clipboardScratch, file->contentStart, (size_t)n);
        clipboardScratch[n] = '\0';
        SetClipboardText(clipboardScratch);
        copyFeedbackTimer = 1.2f;
    }
}

// Shows the most recent assistant reply, split into one or more named
// files (see splitIntoCodeFiles above), with tabs to switch between
// files when there's more than one, and a working copy-to-clipboard
// button for whichever file is currently shown.
static void drawCodePanel(Rectangle area)
{
    DrawRectangleRounded(area, 0.06f, 8, COLOR_BOX_BG);
    DrawRectangleRoundedLines(area, 0.06f, 8, COLOR_BOX_BORDER);

    int count = getChatMessageCount();
    const ChatMessage* lastAssistant = NULL;
    for (int i = count - 1; i >= 0; i--)
    {
        const ChatMessage* msg = getChatMessage(i);
        if (msg->role == ROLE_ASSISTANT) { lastAssistant = msg; break; }
    }

    if (lastAssistant == NULL)
    {
        DrawText("Code", (int)(area.x + 16), (int)(area.y + 14), 16, COLOR_SIDEBAR_TEXT);
        DrawLine((int)area.x, (int)(area.y + 40), (int)(area.x + area.width), (int)(area.y + 40), COLOR_BOX_BORDER);
        DrawText("No code yet.", (int)(area.x + 16), (int)(area.y + 56), 14, COLOR_PLACEHOLDER);
        return;
    }

    static CodeFileRef files[MAX_CODE_FILES];
    static int fileCount = 0;
    static int selectedTab = 0;
    static int lastSeenMessageCount = -1;

    if (count != lastSeenMessageCount)
    {
        fileCount = splitIntoCodeFiles(lastAssistant->text, files, MAX_CODE_FILES);
        selectedTab = 0;
        lastSeenMessageCount = count;
    }

    bool showTabs = fileCount > 1;
    float headerHeight = showTabs ? 68.0f : 40.0f;

    DrawText("Code", (int)(area.x + 16), (int)(area.y + 14), 16, COLOR_SIDEBAR_TEXT);
    Rectangle copyArea = { area.x + area.width - 76.0f, area.y + 8.0f, 60.0f, 24.0f };
    drawCopyButton(copyArea, &files[selectedTab]);
    DrawLine((int)area.x, (int)(area.y + 40), (int)(area.x + area.width), (int)(area.y + 40), COLOR_BOX_BORDER);

    if (showTabs)
    {
        float tabY = area.y + 40.0f;
        float tabWidth = area.width / (float)fileCount;
        Vector2 mouse = GetMousePosition();

        for (int i = 0; i < fileCount; i++)
        {
            Rectangle tabArea = { area.x + i * tabWidth, tabY, tabWidth, 28.0f };
            bool hovering = CheckCollisionPointRec(mouse, tabArea);
            bool selected = (i == selectedTab);

            if (selected) DrawRectangleRec(tabArea, Fade(COLOR_SIDEBAR_HOVER, 0.6f));
            else if (hovering) DrawRectangleRec(tabArea, Fade(COLOR_SIDEBAR_HOVER, 0.3f));

            int fontSize = 12;
            DrawText(files[i].filename, (int)(tabArea.x + 8), (int)(tabArea.y + 8), fontSize,
                     selected ? COLOR_SIDEBAR_TEXT : Fade(COLOR_SIDEBAR_TEXT, 0.6f));

            if (hovering) SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);
            if (hovering && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) selectedTab = i;
        }
        DrawLine((int)area.x, (int)(tabY + 28.0f), (int)(area.x + area.width), (int)(tabY + 28.0f), COLOR_BOX_BORDER);
    }

    const CodeFileRef* shown = &files[selectedTab];

    if (shown->contentLen == 0)
    {
        DrawText("No code yet.", (int)(area.x + 16), (int)(area.y + headerHeight + 8), 14, COLOR_PLACEHOLDER);
        return;
    }

    BeginScissorMode((int)area.x, (int)(area.y + headerHeight), (int)area.width, (int)(area.height - headerHeight - 8));
    int fontSize = 14;
    int lineHeight = 18;
    int y = (int)(area.y + headerHeight + 8);
    int lineStart = 0;
    char lineBuf[128];

    for (int i = 0; i <= shown->contentLen; i++)
    {
        bool atEnd = (i == shown->contentLen);
        if (atEnd || shown->contentStart[i] == '\n')
        {
            int n = i - lineStart;
            if (n > 127) n = 127;
            memcpy(lineBuf, shown->contentStart + lineStart, (size_t)n);
            lineBuf[n] = '\0';
            DrawText(lineBuf, (int)(area.x + 16), y, fontSize, COLOR_TEXT);
            y += lineHeight;
            lineStart = i + 1;
        }
    }
    EndScissorMode();
}

void drawLivePanel(Rectangle area)
{
    // Voice input may have completed a transcription since last frame --
    // send it through the normal message pipeline, then speak the reply
    // back. Deliberately NOT gated behind isTranscribing(): the
    // background transcription thread sets transcriptionReady=true and
    // transcribing=false together, in the same mutex-protected step, so
    // by the time this function next runs, isTranscribing() may already
    // read false even though a result just became available -- gating
    // the poll attempt behind it meant a completed transcription could
    // arrive, sit ready, and never actually get picked up. pollTranscription()
    // itself is cheap (a mutex lock/check/unlock) and already only
    // returns true once, so calling it unconditionally every frame costs
    // nothing and closes that gap.
    {
        char transcript[1024];
        if (pollTranscription(transcript, sizeof(transcript)) && transcript[0] != '\0')
        {
            sendUserMessage(transcript);

            int count = getChatMessageCount();
            if (count > 0)
            {
                const ChatMessage* lastMsg = getChatMessage(count - 1);
                if (lastMsg->role == ROLE_ASSISTANT)
                {
                    if (getLastResponseHadDomainMatch() && getLastResponseDomain() == DOMAIN_CODING)
                    {
                        // A genuine coding match -- reading raw source
                        // code aloud character-by-character is a bad
                        // experience with any TTS, not just this one --
                        // say a short, fixed phrase and let the code
                        // panel (now visible, see showCodePanel below)
                        // show the actual code to read and copy instead.
                        static const char* codeReadyPhrases[] = {
                            "I've written that out -- check the code panel to copy it.",
                            "Here's the code, ready in the panel on the right.",
                            "Done -- take a look at the code panel to grab it.",
                        };
                        static int codePhraseRotation = 0;
                        speak(codeReadyPhrases[codePhraseRotation % 3]);
                        codePhraseRotation++;
                    }
                    else
                    {
                        // Everything else: a genuine math/physics
                        // answer, a greeting reply, or an honest
                        // "outside my range" reply -- engine.c now
                        // generates all of these directly (see
                        // generateResponse()'s !hadMatch handling), so
                        // lastMsg->text is always something sensible to
                        // speak here, not just for genuine matches.
                        speak(lastMsg->text);
                    }
                }
            }
        }
    }

    bool showCodePanel = (getLastResponseDomain() == DOMAIN_CODING) && getLastResponseHadDomainMatch() && (getChatMessageCount() > 0);

    float avatarAreaWidth = showCodePanel ? area.width * 0.55f : area.width;
    Rectangle avatarArea = { area.x, area.y, avatarAreaWidth, area.height };
    Vector2 center = { avatarArea.x + avatarArea.width / 2.0f, avatarArea.y + avatarArea.height / 2.0f - 40.0f };

    const char* status = "Idle";
    float level = 0.0f;
    Color ringColor = COLOR_SEND_ON;

    if (isUserSpeaking())
    {
        // VAD has confirmed this is actually speech, not just ambient
        // sound -- the more vivid color/reaction.
        status = "Listening...";
        level = getMicLevel();
        ringColor = COLOR_SEND_ON;
    }
    else if (isLiveListeningEnabled())
    {
        // Mic is on and VAD is watching, but hasn't confirmed speech yet
        // -- still shows real mic level (so the ring has some life to
        // it as soon as there's any sound) but in a more muted color, so
        // "hearing something" reads differently from "confirmed you're
        // talking".
        status = "Ready...";
        level = getMicLevel();
        ringColor = COLOR_UPLOAD_ICON;
    }
    else if (isTranscribing())
    {
        status = "Thinking...";
        // Transcription is a computation, not a sound -- there's no real
        // level to show here, so this is an honest idle pulse rather
        // than fabricated audio-reactivity.
        level = 0.15f + 0.05f * sinf((float)GetTime() * 3.0f);
        ringColor = COLOR_UPLOAD_ICON;
    }
    else if (isSpeaking())
    {
        status = "Speaking...";
        level = getSpeakingLevel();
        ringColor = COLOR_BIRD_BEAK;
    }
    else
    {
        // Idle placeholder pulse, matching thinking.c's animated avatar
        // technique -- just enough motion that the panel doesn't look
        // frozen while waiting for the next interaction.
        level = 0.12f + 0.04f * sinf((float)GetTime() * 1.5f);
    }

    drawAudioRing(center, 60.0f, smoothRingLevel(level), ringColor);
    drawAnimatedAvatar(center, 60.0f);

    int fontSize = 18;
    int tw = MeasureText(status, fontSize);
    DrawText(status, (int)(center.x - tw / 2.0f), (int)(center.y + 90.0f), fontSize, COLOR_WELCOME_TEXT);

    Vector2 micCenter = { center.x, center.y + 150.0f };
    bool micButtonActive = isLiveListeningEnabled();
    drawMicButton(micCenter, 26.0f, micButtonActive);

    Vector2 mouse = GetMousePosition();
    bool hoveringMic = CheckCollisionPointCircle(mouse, micCenter, 26.0f);
    if (hoveringMic)
        SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);

    if (hoveringMic && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
    {
        // DIAGNOSTIC (temporary): confirms the click itself is being
        // detected, separately from whether startLiveListening() then
        // does anything -- if this never prints when you click the mic,
        // it's a hit-testing/coordinate problem, not a VOICE_ENABLED
        // problem. If it DOES print but isLiveListeningEnabled() stays
        // false right after, listening is compiled out or whisper
        // failed to load (check your console for "[voice] whisper
        // model loaded OK" from voice.c's own diagnostics).
        printf("[live_view] mic clicked, isLiveListeningEnabled before=%d\n", isLiveListeningEnabled());

        // A plain on/off toggle now -- no press-and-release-to-record.
        // VAD decides when you're actually talking; this button just
        // controls whether it's watching at all.
        if (isLiveListeningEnabled())
            stopLiveListening();
        else
            startLiveListening();

        printf("[live_view] after toggle, isLiveListeningEnabled=%d\n", isLiveListeningEnabled());
    }

    if (showCodePanel)
    {
        Rectangle codeArea = { area.x + avatarAreaWidth + 20.0f, area.y + 30.0f, area.width - avatarAreaWidth - 40.0f, area.height - 60.0f };
        drawCodePanel(codeArea);
    }
}