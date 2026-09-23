import Header from "@/app/components/header";
import ChatSection from "./components/chat-section";
import PlansPanel from "./components/plans-panel";

export default function Home() {
  return (
    <main className="min-h-screen w-screen flex justify-center items-start background-gradient py-6">
      <div className="space-y-2 lg:space-y-6 w-[90%] lg:w-[60rem]">
        <Header />
        <div className="h-[65vh] flex">
          <ChatSection />
        </div>
        <PlansPanel />
      </div>
    </main>
  );
}
