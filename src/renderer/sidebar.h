#pragma once

#define SIDEBAR_WIDTH 260.0f
#define ROW_HEIGHT   40.0f
#define ROW_PADDING  14.0f
#define HEADER_HEIGHT 56.0f
#define TAB_ROW_HEIGHT 40.0f
#define MODEL_PICKER_HEIGHT 60.0f

// Which main-content-area view is active. Chat is the default on
// startup; Live is reached via the tab drawSidebar() renders below the
// header.
typedef enum
{
    PANEL_CHAT,
    PANEL_LIVE
} PanelMode;

void drawSidebar(void);
PanelMode getSidebarPanelMode(void);