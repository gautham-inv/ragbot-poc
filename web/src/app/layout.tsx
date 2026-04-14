import "./globals.css";

export const metadata = {
  title: "Gloria Pets Catalog Bot",
  description: "Chat with your product catalog"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
