import Image from "next/image";
import HomeSwitch from "./components/home-switch";

// The chat interface, or the public demo's page when the API runs in public demo mode (components/home-switch.tsx).
export default function Home() {
  return (
    <HomeSwitch
      brand={
        <Image
          className="rounded-lg"
          src="/metanavit.jpeg"
          alt=""
          width={28}
          height={28}
          priority
        />
      }
    />
  );
}
