// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionTicket } from "@/lib/api/types";

const { clients, websocketManagers } = vi.hoisted(() => ({
  clients: [] as Array<{ callbacks: { onUserStartedSpeaking: () => void; onBotStartedSpeaking: () => void; onBotOutput: (d: { text?: string }) => void } }>,
  websocketManagers: [] as Array<{ interruptBotAudio: ReturnType<typeof vi.fn>; bufferBotAudio: ReturnType<typeof vi.fn> }>,
}));

vi.mock("@pipecat-ai/client-js", () => ({
  PipecatClient: class {
    callbacks: (typeof clients)[number]["callbacks"];
    constructor(options: { callbacks: (typeof clients)[number]["callbacks"] }) {
      this.callbacks = options.callbacks;
      clients.push(this);
    }
    connect = vi.fn(async () => undefined);
    disconnect = vi.fn(async () => undefined);
    tracks() {
      return { local: { audio: null } };
    }
  },
}));

vi.mock("@pipecat-ai/small-webrtc-transport", () => ({
  SmallWebRTCTransport: class {
    constructor(_options: unknown) {}
  },
}));

vi.mock("@pipecat-ai/websocket-transport", () => ({
  WavMediaManager: class {
    interruptBotAudio = vi.fn();
    bufferBotAudio = vi.fn();
  },
  WebSocketTransport: class {
    constructor(options: { mediaManager: (typeof websocketManagers)[number] }) {
      websocketManagers.push(options.mediaManager);
    }
  },
}));

import { useVoiceSession } from "@/features/talk/use-voice-session";

const ticket = {
  session_id: "session-1",
  ice_url: "http://ice.local",
  offer_url: "http://offer.local",
  websocket_url: "ws://voice.local/ws",
} as SessionTicket;

function audioElement() {
  return {
    pause: vi.fn(),
    play: vi.fn(() => Promise.resolve()),
    currentTime: 2.4,
  };
}

afterEach(() => {
  clients.length = 0;
  websocketManagers.length = 0;
  vi.unstubAllGlobals();
});

describe("WebRTC barge-in", () => {
  it("pauses the remote audio when the user speaks and resumes it when the bot speaks", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, json: async () => ({ iceServers: [] }) })),
    );
    const { result } = renderHook(() => useVoiceSession(async () => ticket));
    const element = audioElement();
    result.current.audioRef.current = element as unknown as HTMLAudioElement;

    await act(async () => {
      await result.current.start("webrtc");
    });
    const callbacks = clients[0].callbacks;

    act(() => {
      callbacks.onBotStartedSpeaking();
      callbacks.onBotOutput({ text: "Hello there" });
    });
    element.play.mockClear();

    act(() => callbacks.onUserStartedSpeaking());

    expect(element.pause).toHaveBeenCalledOnce();
    expect(element.currentTime).toBe(2.4);
    expect(result.current.state.phase).toBe("user");
    expect(result.current.state.lines.at(-1)).toMatchObject({
      who: "agent",
      text: "Hello there",
      interrupted: true,
      final: true,
    });
    expect(websocketManagers).toHaveLength(0);

    act(() => callbacks.onBotStartedSpeaking());

    expect(element.play).toHaveBeenCalledOnce();
    expect(result.current.audioRef.current).toBe(element);
    expect(result.current.state.phase).toBe("agent");
  });

  it("leaves WebSocket playback on WavMediaManager", async () => {
    const { result } = renderHook(() => useVoiceSession(async () => ticket));
    const element = audioElement();
    result.current.audioRef.current = element as unknown as HTMLAudioElement;

    await act(async () => {
      await result.current.start("websocket");
    });

    expect(websocketManagers).toHaveLength(1);
    const manager = websocketManagers[0];
    act(() => {
      clients[0].callbacks.onUserStartedSpeaking();
      clients[0].callbacks.onBotStartedSpeaking();
    });

    expect(manager.interruptBotAudio).not.toHaveBeenCalled();
    expect(manager.bufferBotAudio).not.toHaveBeenCalled();
    expect(clients).toHaveLength(1);
    expect(result.current.audioRef.current).toBe(element);
  });
});
