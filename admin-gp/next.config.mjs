/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "standalone",
  async rewrites() {
    // In prod, Caddy proxies /api/auth/* to auth-server:4000. In dev there's no
    // Caddy, so we replicate the proxy here so the browser-side Better Auth
    // client (basePath: "/api/auth") can reach the auth-server. AUTH_PROXY_TARGET
    // is read at build time inside the container; defaults to the compose
    // service name which works inside the Docker network.
    const target = process.env.AUTH_PROXY_TARGET || "http://auth-server:4000";
    return [
      {
        source: "/api/auth/:path*",
        destination: `${target}/api/auth/:path*`,
      },
    ];
  },
};

export default nextConfig;
