import "./globals.css";
import UserbackInit from "./UserbackInit";

export const metadata = {
  title: "Gloria Pets Catalog Bot",
  description: "Chat with your product catalog"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <UserbackInit />
        {children}
      </body>
    </html>
  );
}
