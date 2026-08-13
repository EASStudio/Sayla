// Module includes
#include "input.h"
#include "../renderer/colors.h"
#include "chat_history.h"
#include "engine.h"

// Core includes
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
    #define strcasecmp _stricmp
#endif

static int wrapInputText(const char *text, int len, int fontSize, float maxWidth, TextLine lines[], int maxLines)
{
    int lineCount = 0;
    int i = 0;

    if (maxWidth < 1.0f) maxWidth = 1.0f;

    while (i < len && lineCount < maxLines)
    {
        int lineStart = i;
        int end = i;
        int lastSpaceIdx = -1;
        char buf[LINE_MEASURE_BUF];
        int bufLen = 0;

        while (i < len)
        {
            char c = text[i];

            if (bufLen >= LINE_MEASURE_BUF - 1)
            {
                end = i;
                break;
            }

            buf[bufLen] = c;
            buf[bufLen + 1] = '\0';

            float w = (float)MeasureText(buf, fontSize);

            if (w > maxWidth && bufLen > 0)
            {
                end = (lastSpaceIdx >= 0) ? lastSpaceIdx : i;
                break;
            }

            if (c == ' ') lastSpaceIdx = i + 1;

            bufLen++;
            i++;
            end = i;
        }

        lines[lineCount].start = lineStart;
        lines[lineCount].length = end - lineStart;
        lineCount++;
        i = end;
    }

    return lineCount;
}

static void copyLineToBuffer(const char *text, TextLine line, char out[LINE_MEASURE_BUF])
{
    int n = line.length;
    if (n > LINE_MEASURE_BUF - 1) n = LINE_MEASURE_BUF - 1;

    for (int k = 0; k < n; k++) out[k] = text[line.start + k];
    out[n] = '\0';
}

static void drawUploadButton(Vector2 center, float radius, bool hovered, bool active)
{
    Color bg = active ? Fade(COLOR_SEND_ON, 0.25f) : (hovered ? Fade(COLOR_UPLOAD_BG, 0.7f) : COLOR_UPLOAD_BG);
    DrawCircleV(center, radius, bg);
    DrawCircleLines((int)center.x, (int)center.y, radius, Fade(COLOR_UPLOAD_ICON, active ? 0.6f : 0.35f));

    float arm = radius * 0.8f;
    Color icon = active ? COLOR_SEND_ON : COLOR_UPLOAD_ICON;
    DrawLineEx((Vector2){ center.x - arm / 2, center.y }, (Vector2){ center.x + arm / 2, center.y }, 3.0f, icon);
    DrawLineEx((Vector2){ center.x, center.y - arm / 2 }, (Vector2){ center.x, center.y + arm / 2 }, 3.0f, icon);
}

static void drawSendButton(Vector2 center, float radius, bool enabled)
{
    DrawCircleV(center, radius, enabled ? COLOR_SEND_ON : COLOR_SEND_OFF);

    Color icon = WHITE;
    float s = radius * 0.62f;

    Vector2 tip   = { center.x + s * 0.55f, center.y - s * 0.55f };
    Vector2 left  = { center.x - s * 0.55f, center.y + s * 0.15f };
    Vector2 right = { center.x + s * 0.15f, center.y + s * 0.55f };
    Vector2 mid   = { center.x - s * 0.05f, center.y + s * 0.05f };

    DrawTriangle(tip, left, right, icon);
    DrawLineEx(tip, mid, 1.5f, Fade(COLOR_SEND_ON, enabled ? 0.35f : 0.2f));
}

static void extractFileName(const char *path, char *out, int outSize)
{
    const char *last = path;
    for (const char *p = path; *p; p++)
    {
        if (*p == '/' || *p == '\\') last = p + 1;
    }
    strncpy(out, last, outSize - 1);
    out[outSize - 1] = '\0';
}

// Returns true if the extension looks like a plain-text file we can safely read.
static bool isTextFile(const char *name)
{
    const char *ext = strrchr(name, '.');
    if (!ext) return false;
    ext++; // skip the dot

    static const char *textExts[] = 
    {
        "txt", "md", "c", "h", "cpp", "hpp", "py", "js", "ts", "json",
        "csv", "xml", "html", "css", "yml", "yaml", "toml", "ini", "log",
        "sh", "bat", "cmake", "makefile", NULL
    };

    for (int i = 0; textExts[i]; i++)
    {
        if (strcasecmp(ext, textExts[i]) == 0) return true;
    }
    return false;
}

// Reads a text file (capped, so huge files don't stall the UI thread) and
// runs it through the engine's tokenizer to report a token count. Returns
// -1 for non-text files or files that can't be opened.
static int computeAttachmentTokenCount(const char *path, bool isText)
{
    if (!isText) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    static char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    // Strip any embedded nulls so strlen()/tokenizeEncode see the whole read
    for (size_t k = 0; k < n; k++)
        if (buf[k] == '\0') buf[k] = ' ';

    return countTokens(buf);
}

// Build the final user message that includes attachment metadata (and content for small text files).
static void buildMessageWithAttachments(const char *text, Attachment *attachments, int attachmentCount, char *out, size_t outSize)
{
    out[0] = '\0';

    // Start with the typed text
    if (text && text[0])
    {
        strncpy(out, text, outSize - 1);
        out[outSize - 1] = '\0';
    }

    if (attachmentCount <= 0) return;

    // Append a clear attachment section
    size_t used = strlen(out);
    if (used > 0 && used + 2 < outSize)
    {
        out[used++] = '\n';
        out[used++] = '\n';
        out[used] = '\0';
    }

    const char *header = "[Attached files]\n";
    size_t headerLen = strlen(header);
    if (used + headerLen < outSize)
    {
        strcat(out, header);
        used += headerLen;
    }

    for (int i = 0; i < attachmentCount; i++)
    {
        char line[MAX_PATH_LEN + 64];
        snprintf(line, sizeof(line), "- %s (%s)\n", attachments[i].name, attachments[i].path);

        size_t lineLen = strlen(line);
        if (used + lineLen >= outSize - 1) break;
        strcat(out, line);
        used += lineLen;

        // For small text files, also embed a short preview so the model can see the content
        if (isTextFile(attachments[i].name))
        {
            FILE *f = fopen(attachments[i].path, "rb");
            if (f)
            {
                // Read at most ~4 KB so we don't blow the message buffer
                char preview[4096];
                size_t n = fread(preview, 1, sizeof(preview) - 1, f);
                fclose(f);
                preview[n] = '\0';

                // Strip any embedded nulls just in case
                for (size_t k = 0; k < n; k++)
                    if (preview[k] == '\0') preview[k] = ' ';

                if (n > 0 && used + n + 32 < outSize)
                {
                    strcat(out, "  ```\n");
                    used += 5;
                    strncat(out, preview, outSize - used - 8);
                    used = strlen(out);
                    strcat(out, "\n  ```\n");
                    used = strlen(out);
                }
            }
        }
    }
}

static void sendMessage(const char *text, Attachment *attachments, int attachmentCount)
{
    // Build a single message that contains the typed text + attachment info
    // (and content for text files). This keeps the existing chat_history /
    // engine API unchanged while still delivering the files to the model.
    char fullMessage[MAX_INPUT_CHARS + 1];
    buildMessageWithAttachments(text, attachments, attachmentCount,
                                fullMessage, sizeof(fullMessage));

    if (fullMessage[0] == '\0') return;   // nothing to send

    addChatMessage(ROLE_USER, fullMessage);

    char response[MAX_MESSAGE_LENGTH];
    generateResponse(fullMessage, response, sizeof(response));
    addChatMessage(ROLE_ASSISTANT, response);
}

void sendUserMessage(const char *text)
{
    sendMessage(text, NULL, 0);
}

static void buildChipLabel(const char *name, int tokenCount, char *out, size_t outSize)
{
    if (tokenCount >= 0)
        snprintf(out, outSize, "%s (%d tok)", name, tokenCount);
    else
        snprintf(out, outSize, "%s", name);
}

static bool drawAttachmentChip(Rectangle chip, const char *name, Vector2 mouse)
{
    bool hovered = CheckCollisionPointRec(mouse, chip);

    DrawRectangleRounded(chip, 0.4f, 8, hovered ? Fade(COLOR_UPLOAD_BG, 0.9f) : COLOR_UPLOAD_BG);
    DrawRectangleRoundedLines(chip, 0.4f, 8, COLOR_BOX_BORDER);

    int fontSize = 16;
    char label[140];
    strncpy(label, name, sizeof(label) - 1);
    label[sizeof(label) - 1] = '\0';

    int maxTextW = (int)(chip.width - 28);
    while (MeasureText(label, fontSize) > maxTextW && strlen(label) > 4)
    {
        label[strlen(label) - 1] = '\0';
        label[strlen(label) - 3] = '.';
        label[strlen(label) - 2] = '.';
        label[strlen(label) - 1] = '.';
    }

    DrawText(label, (int)(chip.x + 10), (int)(chip.y + (chip.height - fontSize) / 2), fontSize, COLOR_TEXT);

    Rectangle closeBtn = {
        chip.x + chip.width - 24,
        chip.y + 4,
        18,
        chip.height - 8
    };
    bool closeHover = CheckCollisionPointRec(mouse, closeBtn);
    DrawText("×", (int)closeBtn.x + 3, (int)(closeBtn.y + 1), 18,
             closeHover ? COLOR_SEND_ON : COLOR_UPLOAD_ICON);

    return closeHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

void drawInputBox(Rectangle area)
{
    static char input[MAX_INPUT_CHARS + 1] = "\0";
    static int  letterCount = 0;
    static int  frameCounter = 0;
    static int  backspaceHeldFrames = 0;
    static bool mouseClickedOnText = false;

    static Attachment attachments[MAX_ATTACHMENTS];
    static int attachmentCount = 0;
    static bool attachmentMode = false;

    Vector2 mouse = GetMousePosition();
    int fontSize = 20;

    // ---- Accept dropped files ----
    if (IsFileDropped())
    {
        FilePathList dropped = LoadDroppedFiles();
        for (unsigned int i = 0; i < dropped.count && attachmentCount < MAX_ATTACHMENTS; i++)
        {
            bool exists = false;
            for (int j = 0; j < attachmentCount; j++)
            {
                if (strcmp(attachments[j].path, dropped.paths[i]) == 0)
                {
                    exists = true;
                    break;
                }
            }
            if (exists) continue;

            strncpy(attachments[attachmentCount].path, dropped.paths[i], MAX_PATH_LEN - 1);
            attachments[attachmentCount].path[MAX_PATH_LEN - 1] = '\0';
            extractFileName(dropped.paths[i], attachments[attachmentCount].name,
                            sizeof(attachments[attachmentCount].name));
            attachments[attachmentCount].tokenCount = computeAttachmentTokenCount(
                attachments[attachmentCount].path,
                isTextFile(attachments[attachmentCount].name));
            attachmentCount++;
        }
        UnloadDroppedFiles(dropped);
        attachmentMode = false;
    }

    // ---- Layout ----
    float boxX     = area.x + INPUT_BOX_MARGIN;
    float boxWidth = area.width - (INPUT_BOX_MARGIN * 2);

    float sendCenterX   = boxX + boxWidth - BUTTON_RADIUS - BUTTON_GAP;
    float uploadCenterX = boxX + BUTTON_RADIUS + BUTTON_GAP;

    float textAreaX     = uploadCenterX + BUTTON_RADIUS + BUTTON_GAP;
    float textAreaRight = sendCenterX - BUTTON_RADIUS - BUTTON_GAP;
    float textAreaWidth = textAreaRight - textAreaX;

    float chipsHeight = (attachmentCount > 0) ? (CHIP_HEIGHT + CHIP_PAD) : 0.0f;

    TextLine lines[MAX_WRAP_LINES];
    int totalLines = wrapInputText(input, letterCount, fontSize, textAreaWidth, lines, MAX_WRAP_LINES);

    int visibleLines = totalLines;
    if (visibleLines < 1) visibleLines = 1;
    if (visibleLines > MAX_VISIBLE_LINES) visibleLines = MAX_VISIBLE_LINES;

    float paddingV = (INPUT_BOX_HEIGHT - LINE_HEIGHT) / 2.0f;
    float boxHeight = (paddingV * 2.0f) + (visibleLines * LINE_HEIGHT) + chipsHeight;
    if (boxHeight < INPUT_BOX_HEIGHT + chipsHeight) boxHeight = INPUT_BOX_HEIGHT + chipsHeight;

    Rectangle textBox = {
        boxX,
        area.y + area.height - INPUT_BOX_BOTTOM - boxHeight,
        boxWidth,
        boxHeight
    };

    Vector2 sendCenter   = { sendCenterX,   textBox.y + textBox.height - paddingV - (LINE_HEIGHT / 2.0f) };
    Vector2 uploadCenter = { uploadCenterX, textBox.y + textBox.height - paddingV - (LINE_HEIGHT / 2.0f) };

    bool hoveringBox    = CheckCollisionPointRec(mouse, textBox);
    bool hoveringUpload = CheckCollisionPointCircle(mouse, uploadCenter, BUTTON_RADIUS);
    bool hoveringSend   = CheckCollisionPointCircle(mouse, sendCenter, BUTTON_RADIUS);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
    {
        mouseClickedOnText = hoveringBox && !hoveringUpload && !hoveringSend;

        if (hoveringUpload)
            attachmentMode = !attachmentMode;

        if (hoveringSend && (letterCount > 0 || attachmentCount > 0))
        {
            sendMessage(input, attachments, attachmentCount);
            letterCount = 0;
            input[0] = '\0';
            attachmentCount = 0;
            attachmentMode = false;
        }
    }

    if (mouseClickedOnText)
    {
        SetMouseCursor(MOUSE_CURSOR_IBEAM);

        int key = GetCharPressed();
        while (key > 0)
        {
            if ((key >= 32) && (key <= 125) && (letterCount < MAX_INPUT_CHARS))
            {
                input[letterCount] = (char)key;
                input[letterCount + 1] = '\0';
                letterCount++;
            }
            key = GetCharPressed();
        }

        if (IsKeyDown(KEY_BACKSPACE))
        {
            bool shouldDelete = IsKeyPressed(KEY_BACKSPACE) ||
                                 (backspaceHeldFrames > 25 && (backspaceHeldFrames % 3 == 0));

            if (shouldDelete)
            {
                letterCount--;
                if (letterCount < 0) letterCount = 0;
                input[letterCount] = '\0';
            }
            backspaceHeldFrames++;
        }
        else
        {
            backspaceHeldFrames = 0;
        }

        if (IsKeyPressed(KEY_ENTER) && (letterCount > 0 || attachmentCount > 0))
        {
            sendMessage(input, attachments, attachmentCount);
            letterCount = 0;
            input[0] = '\0';
            attachmentCount = 0;
            attachmentMode = false;
        }
    }
    else if (hoveringUpload || hoveringSend)
    {
        SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);
        backspaceHeldFrames = 0;
    }
    else
    {
        SetMouseCursor(MOUSE_CURSOR_DEFAULT);
        backspaceHeldFrames = 0;
    }

    frameCounter = mouseClickedOnText ? frameCounter + 1 : 0;

    // ---- Draw ----
    DrawRectangleRounded(textBox, BOX_ROUNDNESS, BOX_SEGMENTS, COLOR_BOX_BG);
    DrawRectangleRoundedLines(textBox, BOX_ROUNDNESS, BOX_SEGMENTS,
                              mouseClickedOnText ? DARKGRAY : COLOR_BOX_BORDER);

    // Attachment chips
    if (attachmentCount > 0)
    {
        float chipX = textBox.x + CHIP_PAD;
        float chipY = textBox.y + CHIP_PAD;

        for (int i = 0; i < attachmentCount; i++)
        {
            char label[160];
            buildChipLabel(attachments[i].name, attachments[i].tokenCount, label, sizeof(label));

            int labelW = MeasureText(label, 16);
            float chipW = (float)labelW + 36.0f;
            if (chipW < 80.0f) chipW = 80.0f;
            if (chipX + chipW > textBox.x + textBox.width - CHIP_PAD)
                break;

            Rectangle chip = { chipX, chipY, chipW, CHIP_HEIGHT };
            if (drawAttachmentChip(chip, label, mouse))
            {
                for (int j = i; j < attachmentCount - 1; j++)
                    attachments[j] = attachments[j + 1];
                attachmentCount--;
                i--;
            }
            chipX += chipW + CHIP_GAP;
        }
    }

    drawUploadButton(uploadCenter, BUTTON_RADIUS, hoveringUpload, attachmentMode);
    drawSendButton(sendCenter, BUTTON_RADIUS, letterCount > 0 || attachmentCount > 0);

    if (attachmentMode && attachmentCount == 0)
    {
        const char *hint = "Drop files anywhere to attach";
        int hintW = MeasureText(hint, 16);
        int hintX = (int)(textBox.x + (textBox.width - hintW) / 2);
        int hintY = (int)(textBox.y + (textBox.height - 16) / 2);
        DrawText(hint, hintX, hintY, 16, COLOR_PLACEHOLDER);
    }
    else if (letterCount == 0 && !mouseClickedOnText && attachmentCount == 0)
    {
        int textY = (int)(textBox.y + textBox.height / 2 - fontSize / 2);
        DrawText("Write a message...", (int)textAreaX, textY, fontSize, COLOR_PLACEHOLDER);
    }
    else if (letterCount > 0 || mouseClickedOnText)
    {
        int firstVisible = totalLines - MAX_VISIBLE_LINES;
        if (firstVisible < 0) firstVisible = 0;

        char lineBuf[LINE_MEASURE_BUF];
        float textTop = textBox.y + paddingV + chipsHeight;

        for (int idx = firstVisible; idx < totalLines; idx++)
        {
            copyLineToBuffer(input, lines[idx], lineBuf);

            int rowY = (int)(textTop + (idx - firstVisible) * LINE_HEIGHT);
            DrawText(lineBuf, (int)textAreaX, rowY, fontSize, COLOR_TEXT);

            if (idx == totalLines - 1 && mouseClickedOnText && letterCount < MAX_INPUT_CHARS)
            {
                if (((frameCounter / 20) % 2) == 0)
                {
                    int cursorX = (int)textAreaX + MeasureText(lineBuf, fontSize) + 2;
                    DrawText("|", cursorX, rowY, fontSize, COLOR_TEXT);
                }
            }
        }
    }
}