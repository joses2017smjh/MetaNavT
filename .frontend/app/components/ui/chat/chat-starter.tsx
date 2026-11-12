import { useChatUI } from "@llamaindex/chat-ui";
import { StarterQuestions } from "@llamaindex/chat-ui/widgets";
import { useEffect, useState } from "react";
import { apiFetch, useApiKeyVersion } from "./hooks/api-key";
import { useClientConfig } from "./hooks/use-config";

export function ChatStarter() {
  const { append } = useChatUI();
  const { backend } = useClientConfig();
  const [starterQuestions, setStarterQuestions] = useState<string[]>();
  const keyVersion = useApiKeyVersion(); // asked again once a key is entered

  useEffect(() => {
    if (!starterQuestions) {
      apiFetch(`${backend}/api/chat/config`)
        .then((response) => response.json())
        .then((data) => {
          if (data?.starterQuestions) {
            setStarterQuestions(data.starterQuestions);
          }
        })
        .catch((error) => console.error("Error fetching config", error));
    }
  }, [starterQuestions, backend, keyVersion]);

  if (!starterQuestions?.length) return null;
  return <StarterQuestions append={append} questions={starterQuestions} />;
}
