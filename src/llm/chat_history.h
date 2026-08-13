#pragma once

#include <stdbool.h>

#define MAX_CHAT_MESSAGES 500
#define MAX_MESSAGE_LENGTH 4000

typedef enum
{
    ROLE_USER,
    ROLE_ASSISTANT
} ChatRole;

typedef struct
{
    ChatRole role;
    char text[MAX_MESSAGE_LENGTH];
} ChatMessage;

void initChatHistory(void);
void addChatMessage(ChatRole role, const char* text);
int getChatMessageCount(void);
const ChatMessage* getChatMessage(int index);
void clearChatHistory(void);
bool saveChatHistoryToFile(const char* path);