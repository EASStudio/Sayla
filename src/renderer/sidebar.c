// Module includes
#include "sidebar.h"
#include "colors.h"
#include "avatar.h"
#include "../llm/chat_history.h"
#include "../llm/engine.h"

// Core includes
#include <stdio.h>

// Persists between frames so the list stays where the user left it.
static float scrollOffset = 0.0f;
#define SCROLL_SPEED 30.0f

// Defaults to Chat on startup, as requested -- Live is reached by
// clicking the tab below.
static PanelMode currentPanel = PANEL_CHAT;

PanelMode getSidebarPanelMode(void)
{
    return currentPanel;
}

static void drawPanelTabs(Vector2 mouse)
{
    float tabWidth = SIDEBAR_WIDTH / 2.0f;
    Rectangle chatTab = { 0, HEADER_HEIGHT, tabWidth, TAB_ROW_HEIGHT };
    Rectangle liveTab = { tabWidth, HEADER_HEIGHT, tabWidth, TAB_ROW_HEIGHT };

    bool hoveringChat = CheckCollisionPointRec(mouse, chatTab);
    bool hoveringLive = CheckCollisionPointRec(mouse, liveTab);

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
    {
        if (hoveringChat) currentPanel = PANEL_CHAT;
        else if (hoveringLive) currentPanel = PANEL_LIVE;
    }

    Color chatBg = (currentPanel == PANEL_CHAT) ? COLOR_SIDEBAR_HOVER : (hoveringChat ? Fade(COLOR_SIDEBAR_HOVER, 0.5f) : COLOR_SIDEBAR_BG);
    Color liveBg = (currentPanel == PANEL_LIVE) ? COLOR_SIDEBAR_HOVER : (hoveringLive ? Fade(COLOR_SIDEBAR_HOVER, 0.5f) : COLOR_SIDEBAR_BG);

    DrawRectangleRec(chatTab, chatBg);
    DrawRectangleRec(liveTab, liveBg);

    Color chatText = (currentPanel == PANEL_CHAT) ? COLOR_SIDEBAR_TEXT : Fade(COLOR_SIDEBAR_TEXT, 0.6f);
    Color liveText = (currentPanel == PANEL_LIVE) ? COLOR_SIDEBAR_TEXT : Fade(COLOR_SIDEBAR_TEXT, 0.6f);

    int fontSize = 15;
    const char* chatLabel = "Chat";
    const char* liveLabel = "Live";
    int chatW = MeasureText(chatLabel, fontSize);
    int liveW = MeasureText(liveLabel, fontSize);

    DrawText(chatLabel, (int)(chatTab.x + (tabWidth - chatW) / 2), (int)(chatTab.y + (TAB_ROW_HEIGHT - fontSize) / 2), fontSize, chatText);
    DrawText(liveLabel, (int)(liveTab.x + (tabWidth - liveW) / 2), (int)(liveTab.y + (TAB_ROW_HEIGHT - fontSize) / 2), fontSize, liveText);

    if (currentPanel == PANEL_CHAT)
        DrawLine((int)chatTab.x, (int)(chatTab.y + TAB_ROW_HEIGHT - 2), (int)(chatTab.x + tabWidth), (int)(chatTab.y + TAB_ROW_HEIGHT - 2), COLOR_SEND_ON);
    else
        DrawLine((int)liveTab.x, (int)(liveTab.y + TAB_ROW_HEIGHT - 2), (int)(liveTab.x + tabWidth), (int)(liveTab.y + TAB_ROW_HEIGHT - 2), COLOR_SEND_ON);

    if (hoveringChat || hoveringLive)
        SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);

    DrawLine(0, (int)(HEADER_HEIGHT + TAB_ROW_HEIGHT), (int)SIDEBAR_WIDTH, (int)(HEADER_HEIGHT + TAB_ROW_HEIGHT), COLOR_SIDEBAR_BORDER);
}

static bool modelPickerExpanded = false;

static void drawModelPicker(void)
{
    int screenHeight = GetScreenHeight();
    Rectangle footer = { 0, (float)screenHeight - MODEL_PICKER_HEIGHT, SIDEBAR_WIDTH, MODEL_PICKER_HEIGHT };
    Vector2 mouse = GetMousePosition();
    bool hoveringFooter = CheckCollisionPointRec(mouse, footer);
    bool clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);

    if (modelPickerExpanded)
    {
        int modelCount = getModelCount();
        // effort header+3 rows (98) + models header (26) + one row per
        // registered model (26 each) + one fixed "more coming soon" row (26)
        float popupHeight = 98.0f + 26.0f + (float)modelCount * 26.0f + 26.0f;
        Rectangle popup = { 0, footer.y - popupHeight, SIDEBAR_WIDTH, popupHeight };

        DrawRectangleRec(popup, COLOR_SIDEBAR_BG);
        DrawRectangleLinesEx(popup, 1.0f, COLOR_SIDEBAR_BORDER);

        // ---- Effort section: three real, clickable options ----
        DrawText("Effort", (int)ROW_PADDING, (int)(popup.y + 8), 12, Fade(COLOR_SIDEBAR_TEXT, 0.6f));

        EffortLevel levels[3] = { EFFORT_LOW, EFFORT_STANDARD, EFFORT_HIGH };
        EffortLevel current = getEffortLevel();
        bool hoveringAnyRow = false;

        for (int i = 0; i < 3; i++)
        {
            Rectangle row = { 0, popup.y + 26 + (float)i * 24, SIDEBAR_WIDTH, 24 };
            bool hoveringRow = CheckCollisionPointRec(mouse, row);
            bool isCurrent = (levels[i] == current);

            if (isCurrent) DrawRectangleRec(row, Fade(COLOR_SEND_ON, 0.15f));
            else if (hoveringRow) DrawRectangleRec(row, Fade(COLOR_SIDEBAR_HOVER, 0.6f));

            DrawText(effortLevelDisplayName(levels[i]), (int)ROW_PADDING, (int)(row.y + 5), 14,
                     isCurrent ? COLOR_SIDEBAR_TEXT : Fade(COLOR_SIDEBAR_TEXT, 0.75f));
            if (isCurrent)
                DrawText("current", (int)(SIDEBAR_WIDTH - 62), (int)(row.y + 6), 11, Fade(COLOR_SEND_ON, 0.9f));

            if (hoveringRow)
            {
                hoveringAnyRow = true;
                if (clicked) setEffortLevel(levels[i]);
            }
        }

        DrawLine(0, (int)(popup.y + 26 + 72), (int)SIDEBAR_WIDTH, (int)(popup.y + 26 + 72), COLOR_SIDEBAR_BORDER);

        // ---- All models section: driven by the real registry, not a
        // hardcoded name -- registering a new model in engine.c's
        // MODEL_REGISTRY makes it appear here automatically. Selecting
        // one calls setActiveModelIndex(), which blocks while it loads
        // from cache or trains fresh if it's never been trained before
        // (same first-run cost any model pays once) -- for now this
        // runs directly on this thread, so switching to an uncached
        // model will visibly pause the UI; wrapping it in the same kind
        // of background-thread pattern startEngineInitAsync() uses is
        // the natural next step once a second real model actually
        // exists to test that against.
        float modelsY = popup.y + 98;
        DrawText("All models", (int)ROW_PADDING, (int)(modelsY + 8), 12, Fade(COLOR_SIDEBAR_TEXT, 0.6f));

        int activeIdx = getActiveModelIndex();
        for (int i = 0; i < modelCount; i++)
        {
            const ModelConfig* m = getModelConfigAt(i);
            Rectangle row = { 0, modelsY + 26 + (float)i * 26, SIDEBAR_WIDTH, 26 };
            bool hoveringRow = CheckCollisionPointRec(mouse, row);
            bool isCurrent = (i == activeIdx);

            if (isCurrent) DrawRectangleRec(row, Fade(COLOR_SEND_ON, 0.15f));
            else if (hoveringRow) DrawRectangleRec(row, Fade(COLOR_SIDEBAR_HOVER, 0.6f));

            DrawText(m->name, (int)ROW_PADDING, (int)(row.y + 5), 14,
                     isCurrent ? COLOR_SIDEBAR_TEXT : Fade(COLOR_SIDEBAR_TEXT, 0.75f));
            if (isCurrent)
                DrawText("current", (int)(SIDEBAR_WIDTH - 62), (int)(row.y + 6), 11, Fade(COLOR_SEND_ON, 0.9f));

            if (hoveringRow)
            {
                hoveringAnyRow = true;
                if (clicked && !isCurrent) setActiveModelIndex(i);
            }
        }

        Rectangle comingSoonRow = { 0, modelsY + 26 + (float)modelCount * 26, SIDEBAR_WIDTH, 26 };
        DrawText("More coming soon...", (int)ROW_PADDING, (int)(comingSoonRow.y + 5), 13, COLOR_PLACEHOLDER);

        if (hoveringAnyRow) SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);

        // Any click while expanded closes it -- selecting an effort
        // level, selecting a model, or clicking anywhere else all
        // dismiss the popup.
        if (clicked) modelPickerExpanded = false;
    }
    else
    {
        if (hoveringFooter && clicked) modelPickerExpanded = true;
    }

    DrawRectangleRec(footer, COLOR_SIDEBAR_BG);
    DrawLine(0, (int)footer.y, (int)SIDEBAR_WIDTH, (int)footer.y, COLOR_SIDEBAR_BORDER);
    if (hoveringFooter) DrawRectangleRec(footer, Fade(COLOR_SIDEBAR_HOVER, 0.4f));

    DrawText(getModelConfigAt(getActiveModelIndex())->name, (int)ROW_PADDING, (int)(footer.y + 10), 15, COLOR_SIDEBAR_TEXT);
    char effortLabel[32];
    snprintf(effortLabel, sizeof(effortLabel), "Effort: %s", effortLevelDisplayName(getEffortLevel()));
    DrawText(effortLabel, (int)ROW_PADDING, (int)(footer.y + 32), 12, Fade(COLOR_SIDEBAR_TEXT, 0.6f));

    const char* chevron = modelPickerExpanded ? "v" : "^";
    int chevronW = MeasureText(chevron, 14);
    DrawText(chevron, (int)(SIDEBAR_WIDTH - ROW_PADDING - chevronW), (int)(footer.y + (MODEL_PICKER_HEIGHT - 14) / 2), 14, Fade(COLOR_SIDEBAR_TEXT, 0.6f));

    if (hoveringFooter) SetMouseCursor(MOUSE_CURSOR_POINTING_HAND);
}

void drawSidebar(void)
{
    int screenHeight = GetScreenHeight();

    Rectangle panel = { 0, 0, SIDEBAR_WIDTH, (float)screenHeight };
    DrawRectangleRec(panel, COLOR_SIDEBAR_BG);

    // Header: bird avatar + app name (never scrolls)
    drawBirdAvatar((Vector2){ ROW_PADDING + 14, HEADER_HEIGHT / 2 - 6 }, 14.0f);
    DrawText("Sayla", (int)(ROW_PADDING + 36), (int)(HEADER_HEIGHT / 2 - 16), 20, COLOR_SIDEBAR_TEXT);
    DrawLine(0, (int)HEADER_HEIGHT, (int)SIDEBAR_WIDTH, (int)HEADER_HEIGHT, COLOR_SIDEBAR_BORDER);

    Vector2 mouse = GetMousePosition();
    drawPanelTabs(mouse);

    float listTop = HEADER_HEIGHT + TAB_ROW_HEIGHT;

    // Count how many rows we actually have (one per user prompt) so we know
    // whether there's anything to scroll.
    int count = getChatMessageCount();
    int rowCount = 0;
    for (int i = 0; i < count; i++)
    {
        if (getChatMessage(i)->role == ROLE_USER) rowCount++;
    }

    float listHeight = (float)screenHeight - listTop - MODEL_PICKER_HEIGHT;
    float contentHeight = rowCount * ROW_HEIGHT;
    float maxScroll = contentHeight - listHeight;
    if (maxScroll < 0.0f) maxScroll = 0.0f;

    // Scroll only while the mouse is over the message list itself (not the
    // header or the tabs, which CheckCollisionPointRec against the whole
    // panel used to include).
    Rectangle listArea = { 0, listTop, SIDEBAR_WIDTH, listHeight };
    bool hoveringList = CheckCollisionPointRec(mouse, listArea);
    if (hoveringList)
    {
        scrollOffset -= GetMouseWheelMove() * SCROLL_SPEED;
    }

    if (scrollOffset < 0.0f) scrollOffset = 0.0f;
    if (scrollOffset > maxScroll) scrollOffset = maxScroll;

    // Clip the list to the region below the header+tabs so scrolled rows
    // can never draw over them or past the bottom of the window.
    BeginScissorMode(0, (int)listTop, (int)SIDEBAR_WIDTH, (int)listHeight);

    float y = listTop + 14.0f - scrollOffset;

    for (int i = 0; i < count; i++)
    {
        const ChatMessage* msg = getChatMessage(i);
        if (msg->role != ROLE_USER) continue;

        // Skip rows that have scrolled fully out of view
        if (y + ROW_HEIGHT < listTop || y > screenHeight)
        {
            y += ROW_HEIGHT;
            continue;
        }

        char preview[48];
        int n = 0;
        while (msg->text[n] != '\0' && n < 44)
        {
            preview[n] = msg->text[n];
            n++;
        }

        if (msg->text[n] != '\0')
        {
            preview[n++] = '.';
            preview[n++] = '.';
            preview[n++] = '.';
        }
        preview[n] = '\0';

        DrawText(preview, (int)ROW_PADDING, (int)(y + 10), 14, COLOR_SIDEBAR_TEXT);
        y += ROW_HEIGHT;
    }

    EndScissorMode();

    drawModelPicker();
}