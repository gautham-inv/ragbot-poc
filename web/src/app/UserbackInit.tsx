"use client";

import Userback from "@userback/widget";
import { useEffect, useRef } from "react";

type UserbackUser = {
  id: string;
  name?: string;
  email?: string;
};

export default function UserbackInit(props: { user?: UserbackUser }) {
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    const token = process.env.NEXT_PUBLIC_USERBACK_TOKEN;
    if (!token) {
      if (process.env.NODE_ENV !== "production") {
        // eslint-disable-next-line no-console
        console.warn("[Userback] NEXT_PUBLIC_USERBACK_TOKEN not set; widget not initialized");
      }
      return;
    }

    const { user } = props;
    const options = user
      ? {
          user_data: {
            id: user.id,
            info: {
              ...(user.name ? { name: user.name } : {}),
              ...(user.email ? { email: user.email } : {})
            }
          }
        }
      : undefined;

    (async () => {
      try {
        await Userback(token, options as unknown as Record<string, unknown>);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("[Userback] init failed", err);
      }
    })();
  }, [props.user?.email, props.user?.id, props.user?.name]);

  return null;
}
