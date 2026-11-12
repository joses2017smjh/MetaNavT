"use client";

import { ChatSection as ChatSectionUI } from "@llamaindex/chat-ui";
import "@llamaindex/chat-ui/styles/markdown.css";
import "@llamaindex/chat-ui/styles/pdf.css";
import { useChat } from "ai/react";
import CustomChatInput from "./ui/chat/chat-input";
import CustomChatMessages from "./ui/chat/chat-messages";
import { apiFetch } from "./ui/chat/hooks/api-key";
import { useClientConfig } from "./ui/chat/hooks/use-config";
import { ArtifactDownload } from "./ui/chat/tools/artifact-download";

export default function ChatSection() {
  const { backend } = useClientConfig();
  const handler = useChat({
    api: `${backend}/api/chat`,
    fetch: apiFetch, // adds the API key the user entered
    onError: (error: unknown) => {
      if (!(error instanceof Error)) throw error;
      let errorMessage: string;
      try {
        errorMessage = JSON.parse(error.message).detail;
      } catch (e) {
        errorMessage = error.message;
      }
      alert(errorMessage);
    },
  });
  return (
    <>
      <ChatSectionUI handler={handler} className="w-full h-full">
        <CustomChatMessages />
        <CustomChatInput />
      </ChatSectionUI>
      <ArtifactDownload />
    </>
  );
}
