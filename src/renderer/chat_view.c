// Module includes
#include "chat_view.h"
#include "colors.h"
#include "avatar.h"
#include "thinking.h"
#include "../llm/chat_history.h"

// Core includes
#include <string.h>
#include <stdbool.h>

// Word-wraps text to fit maxWidth, recording each line as a (start, length)
// pair rather than copying substrings, same approach as input.c's wrapper.
static int wrapText(const char* text, int fontSize, float maxWidth, int lineStarts[], int lineLens[], int maxLines)
{
    int len = (int)strlen(text);
    int lineCount = 0;
    int i = 0;

    if (maxWidth < 1.0f) maxWidth = 1.0f;

    while (i < len && lineCount < maxLines)
    {
        int lineStart = i;
        int end = i;
        int lastSpace = -1;
        char buf[BUBBLE_MEASURE_BUF];
        int bufLen = 0;
        bool hitNewline = false;

        while (i < len)
        {
            char c = text[i];

            // Explicit newlines force a line break here, matching what
            // DrawText actually does when it renders this same text.
            // Without this, a message containing '\n' (any attachment
            // preview always does -- see buildMessageWithAttachments in
            // input.c) gets wrapped by width alone, undercounting how
            // many visual lines DrawText will actually produce. That
            // mismatch is what let text overflow past the bottom of the
            // bubble: the bubble's height was sized using the undercount.
            if (c == '\n')
            {
                end = i;
                i++; // consume the newline itself, it isn't drawn
                hitNewline = true;
                break;
            }

            if (bufLen >= BUBBLE_MEASURE_BUF - 1)
            {
                end = i;
                break;
            }

            buf[bufLen] = c;
            buf[bufLen + 1] = '\0';

            float w = (float)MeasureText(buf, fontSize);

            if (w > maxWidth && bufLen > 0)
            {
                end = (lastSpace >= 0) ? lastSpace : i;
                break;
            }

            if (c == ' ') lastSpace = i + 1;

            bufLen++;
            i++;
            end = i;
        }

        lineStarts[lineCount] = lineStart;
        lineLens[lineCount] = end - lineStart;
        lineCount++;

        // The newline case already advanced i past the '\n' above --
        // reassigning it to end here would walk it back onto the
        // newline and reprocess it as the start of the next line.
        if (!hitNewline) i = end;
    }

    return lineCount;
}

// Persist across frames so scroll position is kept and we can detect when a
// new message has arrived (to auto-scroll to it).
static float scrollOffset = 0.0f;
static int lastMessageCount = 0;

// Brief "Copied!" feedback after the user clicks a bubble.
static int copyFeedbackFrames = 0;
static Vector2 copyFeedbackPos = { 0 };

void drawChatView(Rectangle area)
{
    int count = getChatMessageCount();
    if (count == 0)
    {
        scrollOffset = 0.0f;
        lastMessageCount = 0;
        copyFeedbackFrames = 0;

        // Even with no messages yet (the very first prompt being
        // processed), the thinking beat should still show -- otherwise
        // the screen would just look blank between send and reply.
        if (isThinking())
        {
            float visible = area.height - CHAT_VIEW_BOTTOM_RESERVED;
            if (visible < 0.0f) visible = 0.0f;
            Vector2 center = { area.x + area.width / 2.0f, area.y + visible / 2.0f };
            drawAnimatedAvatar(center, 28.0f);

            const char* label = currentThinkingLabel();
            int fontSize = 16;
            int tw = MeasureText(label, fontSize);
            DrawText(label, (int)(center.x - tw / 2.0f), (int)(center.y + 44.0f), fontSize, COLOR_WELCOME_TEXT);
        }
        return;
    }

    float maxBubbleWidth = area.width * 0.6f;
    float visibleHeight = area.height - CHAT_VIEW_BOTTOM_RESERVED;
    if (visibleHeight < 0.0f) visibleHeight = 0.0f;

    // First pass: measure every bubble so we know the total transcript
    // height before drawing anything (needed for scroll clamping and
    // auto-scroll-to-latest).
    static float bubbleHeights[MAX_CHAT_MESSAGES];
    float totalHeight = 30.0f; // matches the top margin used below

    for (int i = 0; i < count; i++)
    {
        const ChatMessage* msg = getChatMessage(i);
        int lineStarts[BUBBLE_MAX_LINES];
        int lineLens[BUBBLE_MAX_LINES];
        int lineCount = wrapText(msg->text, BUBBLE_FONT_SIZE, maxBubbleWidth - BUBBLE_PADDING * 2,
                                  lineStarts, lineLens, BUBBLE_MAX_LINES);

        bubbleHeights[i] = BUBBLE_PADDING * 2 + lineCount * BUBBLE_LINE_HEIGHT;
        totalHeight += bubbleHeights[i] + BUBBLE_GAP;
    }

    float maxScroll = totalHeight - visibleHeight;
    if (maxScroll < 0.0f) maxScroll = 0.0f;

    // A new message arrived since last frame -- snap to the bottom so the
    // latest message (and the reply once it lands) is always in view.
    if (count != lastMessageCount)
    {
        scrollOffset = maxScroll;
        lastMessageCount = count;
    }

    // Let the mouse wheel scroll back through history, but only while
    // hovering the transcript itself (not the sidebar or the input box).
    Vector2 mouse = GetMousePosition();
    Rectangle transcriptRegion = { area.x, area.y, area.width, visibleHeight };
    if (CheckCollisionPointRec(mouse, transcriptRegion))
    {
        scrollOffset -= GetMouseWheelMove() * SCROLL_SPEED;
    }

    if (scrollOffset < 0.0f) scrollOffset = 0.0f;
    if (scrollOffset > maxScroll) scrollOffset = maxScroll;

    // Clip the whole transcript to the content area, above the input box --
    // bubbles can never render over the sidebar border or the input box.
    BeginScissorMode((int)area.x, (int)area.y, (int)area.width, (int)visibleHeight);

    float y = area.y + 30.0f - scrollOffset;
    char lineBuf[BUBBLE_MEASURE_BUF];
    bool anyBubbleHovered = false;

    for (int i = 0; i < count; i++)
    {
        const ChatMessage* msg = getChatMessage(i);
        bool isUser = (msg->role == ROLE_USER);
        float bubbleHeight = bubbleHeights[i];

        // Skip bubbles that have scrolled fully out of the visible region
        if (y + bubbleHeight < area.y || y > area.y + visibleHeight)
        {
            y += bubbleHeight + BUBBLE_GAP;
            continue;
        }

        int lineStarts[BUBBLE_MAX_LINES];
        int lineLens[BUBBLE_MAX_LINES];
        int lineCount = wrapText(msg->text, BUBBLE_FONT_SIZE, maxBubbleWidth - BUBBLE_PADDING * 2,
                                  lineStarts, lineLens, BUBBLE_MAX_LINES);

        float bubbleWidth = maxBubbleWidth;
        float bubbleX = isUser
            ? (area.x + area.width - bubbleWidth - BUBBLE_SIDE_MARGIN)
            : (area.x + BUBBLE_SIDE_MARGIN + AVATAR_GUTTER);

        if (!isUser)
        {
            Vector2 avatarCenter = { area.x + BUBBLE_SIDE_MARGIN + 14.0f, y + 14.0f };
            drawBirdAvatar(avatarCenter, 14.0f);
        }

        Rectangle bubble = { bubbleX, y, bubbleWidth, bubbleHeight };

        // --- Click-to-copy interaction ---
        bool hovered = CheckCollisionPointRec(mouse, bubble);
        if (hovered)
        {
            anyBubbleHovered = true;
            // Slight highlight so the user knows the bubble is interactive
            DrawRectangleRounded(bubble, 0.25f, 12,
                                 isUser ? Fade(COLOR_USER_BUBBLE, 0.85f)
                                        : Fade(COLOR_ASSISTANT_BUBBLE, 0.85f));
            DrawRectangleRoundedLines(bubble, 0.25f, 12, COLOR_SEND_ON);
        }
        else
        {
            DrawRectangleRounded(bubble, 0.25f, 12,
                                 isUser ? COLOR_USER_BUBBLE : COLOR_ASSISTANT_BUBBLE);
        }

        if (hovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            SetClipboardText(msg->text);
            copyFeedbackFrames = COPY_FEEDBACK_FRAMES;
            copyFeedbackPos = mouse;
        }

        for (int li = 0; li < lineCount; li++)
        {
            int n = lineLens[li];
            if (n > BUBBLE_MEASURE_BUF - 1) n = BUBBLE_MEASURE_BUF - 1;

            memcpy(lineBuf, msg->text + lineStarts[li], n);
            lineBuf[n] = '\0';

            int rowY = (int)(y + BUBBLE_PADDING + li * BUBBLE_LINE_HEIGHT);
            DrawText(lineBuf, (int)(bubbleX + BUBBLE_PADDING), rowY, BUBBLE_FONT_SIZE, COLOR_TEXT);
        }

        y += bubbleHeight + BUBBLE_GAP;
    }

    EndScissorMode();

    // Change cursor when hovering any bubble
    if (anyBubbleHovered)
        SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);

    // --- "Copied!" feedback ---
    if (copyFeedbackFrames > 0)
    {
        const char* label = "Copied!";
        int fontSize = 16;
        int tw = MeasureText(label, fontSize);
        float pad = 8.0f;

        Rectangle toast = {
            copyFeedbackPos.x - tw / 2.0f - pad,
            copyFeedbackPos.y - 36.0f,
            (float)tw + pad * 2.0f,
            28.0f
        };

        // Keep the toast inside the content area
        if (toast.x < area.x + 8) toast.x = area.x + 8;
        if (toast.x + toast.width > area.x + area.width - 8)
            toast.x = area.x + area.width - 8 - toast.width;

        DrawRectangleRounded(toast, 0.3f, 8, Fade(COLOR_SEND_ON, 0.95f));
        DrawText(label,
                 (int)(toast.x + pad),
                 (int)(toast.y + (toast.height - fontSize) / 2),
                 fontSize, WHITE);

        copyFeedbackFrames--;
    }

    // --- "Thinking" indicator, shown near the bottom while a reply is
    // being prepared. Kept as a fixed overlay (not part of the
    // scrollable transcript above) so it doesn't need folding into the
    // scroll/height bookkeeping at the top of this function.
    if (isThinking())
    {
        Vector2 avatarCenter = { area.x + BUBBLE_SIDE_MARGIN + 14.0f, area.y + visibleHeight - 30.0f };
        drawAnimatedAvatar(avatarCenter, 14.0f);

        const char* label = currentThinkingLabel();
        DrawText(label, (int)(avatarCenter.x + 26.0f), (int)(avatarCenter.y - 8.0f), 15, COLOR_WELCOME_TEXT);
    }
}