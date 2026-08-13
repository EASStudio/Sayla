// Module includes
#include "welcome.h"
#include "colors.h"
 
const char* welcomeMessages[] = { "I'm ready when you are", "How can I help?", "Ready to dive in?", "Whats on you mind today?", "Where should we begin?", "Welcome, how may I assist you?" };
const int welcomeMessagesCount = sizeof(welcomeMessages) / sizeof(welcomeMessages[0]);
 
void drawWelcomeBar(Rectangle area)
{
    static int chosenIndex = -1;
    if (chosenIndex < 0)
    {
        chosenIndex = GetRandomValue(0, welcomeMessagesCount - 1);
    }
 
    const char *message = welcomeMessages[chosenIndex];
    int fontSize = 18;
    int textWidth = MeasureText(message, fontSize);

    float centerX = area.x + area.width / 2.0f;
    float centerY = area.y + area.height / 2.0f;
 
    DrawText(message, (int)(centerX - textWidth / 2), (int)(centerY + 20), fontSize, COLOR_WELCOME_TEXT);
}