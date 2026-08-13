// Module includes
#include "avatar.h"
#include "colors.h"

void drawBirdAvatar(Vector2 center, float radius)
{
    // --- Raised wings (drawn first so body sits on top) ---
    // Broader, more goose-like wings that sweep up and out
    Vector2 lBase  = { center.x - radius * 0.22f, center.y + radius * 0.08f };
    Vector2 lTip   = { center.x - radius * 1.05f, center.y - radius * 0.95f };
    Vector2 lOuter = { center.x - radius * 1.25f, center.y - radius * 0.10f };
    DrawTriangle(lBase, lTip, lOuter, COLOR_BIRD_WING);

    Vector2 rBase  = { center.x + radius * 0.22f, center.y + radius * 0.08f };
    Vector2 rTip   = { center.x + radius * 1.05f, center.y - radius * 0.95f };
    Vector2 rOuter = { center.x + radius * 1.25f, center.y - radius * 0.10f };
    DrawTriangle(rBase, rTip, rOuter, COLOR_BIRD_WING);

    // Inner wing highlights (lighter sky blue)
    Color wingHighlight = (Color){
        (unsigned char)(COLOR_BIRD_WING.r + 30),
        (unsigned char)(COLOR_BIRD_WING.g + 30),
        (unsigned char)(COLOR_BIRD_WING.b + 25),
        255
    };
    DrawTriangle(lBase,
                 (Vector2){ center.x - radius * 0.60f, center.y - radius * 0.55f },
                 (Vector2){ center.x - radius * 0.78f, center.y + radius * 0.02f },
                 wingHighlight);

    DrawTriangle(rBase,
                 (Vector2){ center.x + radius * 0.60f, center.y - radius * 0.55f },
                 (Vector2){ center.x + radius * 0.78f, center.y + radius * 0.02f },
                 wingHighlight);

    // --- Torso: longer, more horizontal classic goose body ---
    Vector2 torso = { center.x - radius * 0.05f, center.y + radius * 0.32f };
    float torsoW = radius * 0.85f;   // wider / longer for goose
    float torsoH = radius * 0.52f;   // flatter
    DrawEllipse((int)torso.x, (int)torso.y, torsoW, torsoH, COLOR_BIRD_BODY);

    // Soft belly highlight
    DrawEllipse((int)torso.x, (int)(torso.y + radius * 0.10f),
                radius * 0.48f, radius * 0.28f, Fade(WHITE, 0.22f));

    // --- Long S-curved goose neck (two overlapping ellipses) ---
    // Lower neck segment (rising from body)
    DrawEllipse((int)(center.x + radius * 0.08f),
                (int)(center.y + radius * 0.02f),
                radius * 0.20f, radius * 0.28f, COLOR_BIRD_BODY);

    // Upper neck segment (curving forward toward head)
    DrawEllipse((int)(center.x + radius * 0.22f),
                (int)(center.y - radius * 0.28f),
                radius * 0.18f, radius * 0.30f, COLOR_BIRD_BODY);

    // --- Head (smaller, more forward – classic goose profile) ---
    Vector2 head = { center.x + radius * 0.38f, center.y - radius * 0.52f };
    float headR = radius * 0.32f;
    DrawCircleV(head, headR, COLOR_BIRD_BODY);

    // Subtle head highlight
    DrawCircleV((Vector2){ head.x - headR * 0.18f, head.y - headR * 0.22f },
                headR * 0.50f, Fade(WHITE, 0.15f));

    // Black beak
    float billBaseX = head.x + headR * 0.55f;
    float billLen = headR * 1.35f;
    if (billLen < 9.0f) billLen = 9.0f;
    float billBaseHalfHeight = headR * 0.38f;
    if (billBaseHalfHeight < 3.2f) billBaseHalfHeight = 3.2f;
    // Tip almost comes to a point (very small half-height)
    float billTipHalfHeight = billBaseHalfHeight * 0.18f;
    if (billTipHalfHeight < 1.1f) billTipHalfHeight = 1.1f;
    // Almost no droop -- chisels are straight
    float billDroop = billLen * 0.02f;

    Vector2 billBaseTop    = { billBaseX, head.y - billBaseHalfHeight };
    Vector2 billBaseMid    = { billBaseX, head.y };
    Vector2 billBaseBottom = { billBaseX, head.y + billBaseHalfHeight };
    Vector2 billTipTop     = { billBaseX + billLen, head.y - billTipHalfHeight + billDroop };
    Vector2 billTipMid     = { billBaseX + billLen, head.y + billDroop };
    Vector2 billTipBottom  = { billBaseX + billLen, head.y + billTipHalfHeight + billDroop };

    // Solid fill with the defined beak color
    DrawTriangle(billBaseTop, billTipTop, billTipMid, COLOR_BIRD_BEAK);
    DrawTriangle(billBaseTop, billTipMid, billBaseMid, COLOR_BIRD_BEAK);
    DrawTriangle(billBaseMid, billTipMid, billTipBottom, COLOR_BIRD_BEAK);
    DrawTriangle(billBaseMid, billTipBottom, billBaseBottom, COLOR_BIRD_BEAK);

    // Black midline to show the beak opening (mandible split)
    DrawLineEx(billBaseMid, billTipMid, 1.4f, BLACK);

    // --- Eye ---
    Vector2 eye = { head.x + headR * 0.05f, head.y - headR * 0.15f };
    DrawCircleV(eye, headR * 0.28f, WHITE);
    DrawCircleV(eye, headR * 0.14f, BLACK);
    // Specular highlight
    DrawCircleV((Vector2){ eye.x - headR * 0.06f, eye.y - headR * 0.06f },
                headR * 0.06f, WHITE);
}