import Header from "@/app/components/header";
import HomeSwitch from "./components/home-switch";

// The chat interface, or the public demo's page when the API runs in public demo mode (components/home-switch.tsx).
export default function Home() {
  return (
    <main className="min-h-screen w-screen flex justify-center items-start background-gradient py-6">
      <div className="space-y-2 lg:space-y-6 w-[90%] lg:w-[60rem]">
        <Header />
        <HomeSwitch />
      </div>
    </main>
  );
}
