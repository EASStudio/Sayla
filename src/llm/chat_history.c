// Module includes
#include "chat_history.h"

// Core includes
#include <stddef.h>
#include <stdio.h>

static ChatMessage messages[MAX_CHAT_MESSAGES];
static int messageCount = 0;

void initChatHistory(void)
{
    messageCount = 0;
}

void addChatMessage(ChatRole role, const char* text)
{
    if (messageCount >= MAX_CHAT_MESSAGES) return; // TODO: evict oldest once history overflows

    messages[messageCount].role = role;

    int n = 0;
    while (text[n] != '\0' && n < MAX_MESSAGE_LENGTH - 1)
    {
        messages[messageCount].text[n] = text[n];
        n++;
    }
    messages[messageCount].text[n] = '\0';

    messageCount++;
}

int getChatMessageCount(void)
{
    return messageCount;
}

const ChatMessage* getChatMessage(int index)
{
    if (index < 0 || index >= messageCount) return NULL;
    return &messages[index];
}

void clearChatHistory(void)
{
    messageCount = 0;
}

bool saveChatHistoryToFile(const char* path)
{
    FILE* f = fopen(path, "w");
    if (!f) return false;

    for (int i = 0; i < messageCount; i++)
    {
        const char* roleLabel = (messages[i].role == ROLE_USER) ? "User" : "Assistant";
        fprintf(f, "%s:\n%s\n\n", roleLabel, messages[i].text);
    }

    fclose(f);
    return true;
}